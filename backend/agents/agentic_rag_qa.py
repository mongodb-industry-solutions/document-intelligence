"""
Agentic RAG Q&A System
Implements LangGraph-based agentic retrieval-augmented generation with self-correction
"""

import logging
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.mongodb.base import MongoDBStore

from agents.state import SearchResult
from tools.embedding_tools import generate_query_embedding_direct

import os 
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentGradingResult(BaseModel):
    """Result of document relevance grading"""
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")
    confidence: float = Field(description="Confidence in the grading decision (0.0-1.0)")
    reasoning: str = Field(description="Explanation for the grading decision")


class AgenticRAGResponse(BaseModel):
    """Enhanced response model for agentic RAG"""
    answer: str
    source_chunks: List[str]
    source_documents: List[str]
    confidence: float
    reasoning: str
    citations: Optional[List[Dict[str, Any]]] = None
    workflow_steps: List[str] = Field(default_factory=list)  # Track which nodes were executed
    grading_results: Optional[List[DocumentGradingResult]] = None
    query_rewrites: Optional[List[str]] = None
    session_id: Optional[str] = None  # Session ID for memory tracking


class QueryGeneratorNode:
    """
    Node that decides whether to retrieve documents or respond directly.
    Implements the first decision point in agentic RAG.
    """
    
    def __init__(self, llm: ChatBedrock, tools, mongodb_connector=None):
        """
        Initialize Query Generator Node.
        
        Args:
            llm: Language model for decision making
            tools: List of tools (retriever_tool, report_dates_tool, etc.)
            mongodb_connector: MongoDB connector for persona retrieval
        """
        self.llm = llm
        self.tools = tools if isinstance(tools, list) else [tools]  # Handle both single tool and list
        self.retriever_tool = self.tools[0] if self.tools else None  # Keep for backward compatibility
        self.mongodb_connector = mongodb_connector
        self.name = "generate_query_or_respond"
        self._current_use_case = None  # Store current use case for persona lookup
        
    def set_use_case(self, use_case: str):
        """Set the current use case for persona lookup."""
        self._current_use_case = use_case
    
    def _get_persona_config(self) -> Dict[str, Any]:
        """
        Get persona configuration based on current use case.
        
        Returns:
            Dictionary with persona configuration
        """
        # Try to get persona from MongoDB
        if self._current_use_case and self.mongodb_connector:
            try:
                # For FSI use cases, use 'fsi' as industry
                # This can be made more flexible if needed
                persona_doc = self.mongodb_connector.get_agent_persona("fsi", self._current_use_case)
                
                if persona_doc and "agent_config" in persona_doc:
                    return persona_doc["agent_config"]
            except Exception as e:
                logger.warning(f"Error fetching persona config: {e}")
        
        # Return default persona
        return {
            "capabilities_intro": "I am an AI Document Intelligence Assistant that can help you with document analysis:",
            "capabilities": [
                "Answer questions about document content",
                "Summarize documents and extract insights",
                "Compare information across documents",
                "Identify patterns and relationships in document data"
            ],
            "persona_name": "Document Assistant"
        }
    
    def __call__(self, state: MessagesState) -> Dict[str, Any]:
        """
        Generate query or respond directly based on user input.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with AI response (potentially with tool calls)
        """
        logger.info("🤖 Generate Query or Respond: Analyzing user input for retrieval decision")
        
        # Get persona configuration
        persona = self._get_persona_config()
        
        # Build capabilities list
        capabilities_text = "\n".join([f"{i+1}. {cap}" for i, cap in enumerate(persona["capabilities"])])
        
        # Create system message with dynamic persona
        system_message = {
            "role": "system",
            "content": f"""{persona['capabilities_intro']}

{capabilities_text}

CRITICAL TOOL USAGE INSTRUCTIONS:

Available Tools:
1. retrieve_documents: Search and retrieve information from document content
2. check_latest_report_date: Check when reports were generated for specific industries/use cases

You MUST use the retrieve_documents tool for:
- Questions about specific metrics, numbers, ratings, or data (e.g., "What is the credit rating?", "What are the leverage metrics?")
- Questions starting with "What is/are the...", "How much...", "When did...", "What factors..."
- ANY question asking for information FROM the documents
- Questions about document content, facts, or specific details
- Even if the question seems general, if it's about document content, USE THE TOOL

You MUST use the check_latest_report_date tool for:
- Questions about when reports were generated or created
- Questions about the date of the last/latest report
- Questions about report generation timestamps
- Questions about which reports are available

ONLY respond directly WITHOUT tools for:
- "What can you do for me?" - Introduce yourself as {persona['persona_name']} and list your capabilities above
- "What questions have I asked you so far?" - Review conversation history only
- General greetings or small talk

IMPORTANT RESPONSE GUIDELINES:
- If you cannot answer a question based on the available documents or tools, politely state: "I'm unable to answer that question based on the selected documents and available information. Please try rephrasing your question or asking about content within the documents."
- NEVER mention SQL, databases, system administrators, or technical implementation details
- NEVER suggest users should check logs, databases, or contact administrators
- This is a MongoDB-based system - avoid any references to SQL or relational databases
- Keep responses focused on what you CAN help with based on the documents

DEFAULT BEHAVIOR: When in doubt, USE the appropriate tool. 
It's better to search and find nothing than to ask for clarification when documents are selected.

Be helpful, accurate, and provide expertise in {persona.get('specialization', 'document analysis')}."""
        }
        
        # Add system message to the conversation
        messages_with_system = [system_message] + state["messages"]
        
        # Bind all tools to the LLM
        response = (
            self.llm
            .bind_tools(self.tools)
            .invoke(messages_with_system)
        )
        
        logger.info(f"🤖 LLM Response: {response.content[:200]}...")
        logger.info(f"🤖 Tool calls: {getattr(response, 'tool_calls', [])}")
        
        return {"messages": [response]}


class DocumentGraderNode:
    """
    Node that grades retrieved documents for relevance to the user's question.
    Implements quality control in the agentic RAG pipeline.
    """
    
    def __init__(self, llm: ChatBedrock, mongodb_connector=None):
        """
        Initialize Document Grader Node.
        
        Args:
            llm: Language model for document grading
            mongodb_connector: MongoDB connector for storing grading results
        """
        self.llm = llm
        self.mongodb_connector = mongodb_connector
        self.name = "grade_documents"
        
        # Grading prompt for document relevance assessment
        self.grade_prompt = (
            "You are a grader assessing relevance of retrieved documents to a user question.\n"
            "Here is the retrieved document content:\n\n{context}\n\n"
            "Here is the user question: {question}\n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
            "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.\n"
            "Also provide your confidence level (0.0-1.0) and reasoning for the decision."
        )
    
    def __call__(self, state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """
        Grade retrieved documents for relevance.
        
        Args:
            state: Current conversation state with retrieved documents
            
        Returns:
            Next node to execute based on grading result
        """
        logger.info("🔍 Grade Documents: Assessing relevance of retrieved content")
        
        # Extract the most recent user question and context from state
        user_messages = [msg for msg in state["messages"] if hasattr(msg, 'content') and isinstance(msg, HumanMessage)]
        if user_messages:
            question = user_messages[-1].content  # Get the most recent user question
        else:
            question = state["messages"][0].content  # Fallback to first message
        context = state["messages"][-1].content
        
        # Create grading prompt
        prompt = self.grade_prompt.format(question=question, context=context)
        
        # Get structured output for grading
        response = (
            self.llm
            .with_structured_output(DocumentGradingResult)
            .invoke([{"role": "user", "content": prompt}])
        )
        
        # Store grading result in MongoDB
        self._store_grading_result(question, context, response)
        
        # Route based on grading result
        if response.binary_score == "yes":
            logger.info("✅ Documents deemed relevant - proceeding to answer generation")
            return "generate_answer"
        else:
            logger.info("❌ Documents deemed irrelevant - triggering query rewrite")
            return "rewrite_question"
    
    def _store_grading_result(self, question: str, context: str, result: DocumentGradingResult):
        """Store document grading result in MongoDB assessments collection."""
        try:
            if not self.mongodb_connector:
                logger.debug("No MongoDB connector available for storing grading result")
                return
                
            assessment_doc = {
                "question": question,
                "context_preview": context[:500],  # Store preview for debugging
                "grading_result": result.model_dump(),
                "graded_at": datetime.now(timezone.utc),
                "node": "document_grader"
            }
            
            # Store in gradings collection (QA-specific)
            self.mongodb_connector.gradings_collection.insert_one(assessment_doc)
            
            logger.debug(f"Stored grading result: {result.binary_score} (confidence: {result.confidence})")
            
        except Exception as e:
            logger.warning(f"Failed to store grading result: {e}")


class QueryRewriterNode:
    """
    Node that rewrites user queries to improve retrieval results.
    Implements self-correction mechanism in agentic RAG.
    """
    
    def __init__(self, llm: ChatBedrock):
        """
        Initialize Query Rewriter Node.
        
        Args:
            llm: Language model for query rewriting
        """
        self.llm = llm
        self.name = "rewrite_question"
        
        # Query rewriting prompt
        self.rewrite_prompt = (
            "You are a query improvement specialist. The user asked: \"{question}\"\n"
            "The retrieved documents were not relevant to this question.\n"
            "Rewrite the question to be more specific, clear, and likely to retrieve relevant information.\n"
            "Focus on:\n"
            "1. Making the question more specific\n"
            "2. Using domain-appropriate terminology\n"
            "3. Breaking down complex questions into simpler parts\n"
            "4. Adding context that might help with retrieval\n\n"
            "Provide only the improved question, nothing else."
        )
    
    def __call__(self, state: MessagesState) -> Dict[str, Any]:
        """
        Rewrite the user's question to improve retrieval.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with rewritten question
        """
        logger.info("✏️ Rewrite Question: Improving question for better retrieval")
        
        # Extract the most recent user question (not the first message)
        user_messages = [msg for msg in state["messages"] if hasattr(msg, 'content') and isinstance(msg, HumanMessage)]
        if user_messages:
            question = user_messages[-1].content  # Get the most recent user question
        else:
            question = state["messages"][0].content  # Fallback to first message
        
        # Create rewriting prompt
        prompt = self.rewrite_prompt.format(question=question)
        
        # Generate improved question
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        improved_question = response.content.strip()
        
        logger.info(f"Original: {question}")
        logger.info(f"Rewritten: {improved_question}")
        
        # Return new user message with improved question
        return {"messages": [HumanMessage(content=improved_question)]}


class AnswerGeneratorNode:
    """
    Node that generates final answers based on relevant retrieved documents.
    Implements the answer synthesis in agentic RAG.
    """
    
    def __init__(self, llm: ChatBedrock):
        """
        Initialize Answer Generator Node.
        
        Args:
            llm: Language model for answer generation
        """
        self.llm = llm
        self.name = "generate_answer"
        
        # Answer generation prompt
        self.answer_prompt = (
            "You are an assistant for question-answering tasks.\n"
            "Use the following pieces of retrieved context to answer the question.\n"
            "If you don't know the answer, just say that you don't know.\n"
            "Provide a comprehensive answer based on the context.\n\n"
            "IMPORTANT FORMATTING RULES:\n"
            "- Start DIRECTLY with the answer - do NOT use phrases like 'Based on the provided context', 'According to the documents', 'Based on the information', etc.\n"
            "- Do NOT include citations or references to specific documents or chunks in your answer.\n"
            "- Write naturally as if the information is your knowledge.\n"
            "- Be comprehensive but concise.\n"
            "- Use clear structure with bullet points or paragraphs as appropriate.\n\n"
            "Question: {question}\n"
            "Context: {context}\n\n"
            "Provide your answer now (start directly with the substantive content):"
        )
    
    def __call__(self, state: MessagesState) -> Dict[str, Any]:
        """
        Generate final answer based on relevant context.
        
        Args:
            state: Current conversation state with relevant documents
            
        Returns:
            Updated state with final answer
        """
        logger.info("📝 Generate Answer: Synthesizing final answer")
        
        # Extract the most recent user question and context
        user_messages = [msg for msg in state["messages"] if hasattr(msg, 'content') and isinstance(msg, HumanMessage)]
        if user_messages:
            question = user_messages[-1].content  # Get the most recent user question
        else:
            question = state["messages"][0].content  # Fallback to first message
        context = state["messages"][-1].content
        
        # Create answer generation prompt
        prompt = self.answer_prompt.format(question=question, context=context)
        
        # Generate answer
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        
        return {"messages": [response]}


class AgenticRAGQandA:
    """
    Main Agentic RAG Q&A System.
    
    Implements the complete agentic RAG workflow with:
    - Conditional routing (retrieve vs respond directly)
    - Document grading for relevance assessment
    - Query rewriting for self-correction
    - Answer generation with citations
    
    Architecture follows the LangGraph Agentic RAG pattern with MongoDB integration.
    """
    
    def __init__(
        self,
        llm: ChatBedrock = None,
        embeddings_client=None,
        mongodb_connector=None,
        bedrock_client=None,
        checkpointer=None
    ):
        """
        Initialize Agentic RAG Q&A System.
        
        Args:
            llm: Language model for all nodes
            embeddings_client: VoyageAI embeddings client
            mongodb_connector: MongoDB connector for chunks collection access
            bedrock_client: AWS Bedrock client
            checkpointer: MongoDB checkpointer for conversation persistence
        """
        logger.info("🔧 Building Agentic RAG workflow")
        # Initialize LLM if not provided
        if not llm:
            from cloud.aws.bedrock.client import BedrockClient
            if not bedrock_client:
                bedrock_client = BedrockClient()._get_bedrock_client()
            self.llm = ChatBedrock(
                model=os.getenv("BEDROCK_MODEL_ID"),
                client=bedrock_client,
                provider="anthropic",
                temperature=0.0001,
                max_tokens=2048,  # Limit response length for faster generation
                model_kwargs={
                    "max_tokens": 2048,
                    "temperature": 0.0001
                }
            )
        else:
            self.llm = llm
            
        self.embeddings_client = embeddings_client
        self.mongodb_connector = mongodb_connector
        self.checkpointer = checkpointer
        self.name = "AgenticRAGQandA"
        
        # Initialize tools
        self.retriever_tool = self._create_retriever_tool()
        self.report_dates_tool = self._create_report_dates_tool()
        self.tools = [self.retriever_tool, self.report_dates_tool]
        
        # Initialize nodes with both tools
        self.query_generator = QueryGeneratorNode(self.llm, self.tools, self.mongodb_connector)
        self.document_grader = DocumentGraderNode(self.llm, self.mongodb_connector)
        self.query_rewriter = QueryRewriterNode(self.llm)
        self.answer_generator = AnswerGeneratorNode(self.llm)
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        logger.info("🎯 Agentic RAG Q&A System initialized")
    
    def _create_retriever_tool(self):
        """Create retriever tool using MongoDB connector's vector search method."""
        from langchain.tools import tool
        
        @tool
        def retrieve_documents(query: str) -> str:
            """Search and return information from the document knowledge base using MongoDB connector's vector search."""
            try:
                # Generate query embedding
                embedding_result = generate_query_embedding_direct(
                    query=query,
                    embeddings_client=self.embeddings_client
                )
                
                if not embedding_result["success"]:
                    return "Error: Failed to generate query embedding"
                
                # Apply document filtering if selected_documents are provided
                mongo_filter = None
                if hasattr(self, '_selected_documents') and self._selected_documents:
                    mongo_filter = {"document_id": {"$in": self._selected_documents}}
                    logger.info(f"🔍 Filtering vector search by selected documents: {self._selected_documents}")
                
                # Use MongoDB connector's vector search method
                # Increase k to ensure we get content from all selected documents
                k_value = 10 if mongo_filter else 5  # More results when filtering by documents
                search_results = self.mongodb_connector.vector_search(
                    query_embedding=embedding_result["query_embedding"],
                    k=k_value,
                    filter=mongo_filter
                )
                
                if not search_results:
                    return "No relevant documents found"
                
                # Format results with rich metadata from chunks
                formatted_results = []
                source_citations = []
                
                for i, result in enumerate(search_results):
                    chunk_metadata = result.get('metadata', {})
                    chunk_info = chunk_metadata.get('chunk_metadata', {})
                    
                    source_info = f"[Source: {result.get('document_name', 'Unknown')}]"
                    if chunk_info.get('chunk_index') is not None:
                        source_info += f" [Chunk {chunk_info['chunk_index'] + 1}/{chunk_info.get('total_chunks', '?')}]"
                    
                    # Add similarity score if available
                    if 'similarity_score' in result:
                        source_info += f" [Score: {result['similarity_score']:.3f}]"
                    
                    formatted_results.append(
                        f"{source_info}\n"
                        f"{result.get('chunk_text', '')}\n"
                    )
                    
                    # Create citation for source modal
                    citation = {
                        "id": f"citation_{i+1}",
                        "document_id": result.get('document_id'),
                        "document_name": result.get('document_name'),
                        "chunk_text": result.get('chunk_text'),
                        "chunk_index": chunk_info.get('chunk_index'),
                        "total_chunks": chunk_info.get('total_chunks'),
                        "similarity_score": result.get('similarity_score'),
                        "section_title": result.get('section_title'),
                        "contains_images": chunk_info.get('contains_images', False),
                        "metadata": {
                            **chunk_metadata,
                            "_id": str(result.get('_id'))  # Add chunk MongoDB ID
                        }
                    }
                    source_citations.append(citation)
                
                # Store citations for later use in response
                self._last_citations = source_citations
                
                return "\n---\n".join(formatted_results)
                
            except Exception as e:
                logger.error(f"Error in custom retriever: {e}")
                return f"Error retrieving documents: {str(e)}"
        
        return retrieve_documents
    
    def _create_report_dates_tool(self):
        """Create a tool for checking latest report dates by industry and use case."""
        from langchain.tools import tool
        
        @tool
        def check_latest_report_date(industry: str = None, use_case: str = None) -> str:
            """
            Check the latest report generation date for a specific industry and/or use case.
            
            Args:
                industry: Optional industry code (e.g., 'fsi', 'healthcare')
                use_case: Optional use case code (e.g., 'credit_rating', 'investment_research')
                
            Returns:
                Information about the latest report date(s)
            """
            try:
                # Build query filter
                filter_query = {"status": "generated"}
                if industry:
                    filter_query["industry"] = industry.lower()
                if use_case:
                    filter_query["use_case"] = use_case.lower()
                
                # Query scheduled reports collection - get only the most recent one
                reports = list(self.mongodb_connector.scheduled_reports_collection.find(
                    filter_query,
                    {"industry": 1, "use_case": 1, "report_date": 1, "generated_at": 1}
                ).sort("report_date", -1).limit(1))
                
                if not reports:
                    if industry and use_case:
                        return f"No reports found for industry '{industry}' and use case '{use_case}'."
                    elif industry:
                        return f"No reports found for industry '{industry}'."
                    elif use_case:
                        return f"No reports found for use case '{use_case}'."
                    else:
                        return "No reports found in the system."
                
                # Format response - always show only the most recent report
                most_recent = reports[0]
                report_date = most_recent['report_date'].strftime('%B %d, %Y')
                generated_at = most_recent['generated_at'].strftime('%B %d, %Y at %I:%M %p UTC')
                
                # Build response based on query specificity
                if industry and use_case:
                    return f"The latest report for {most_recent['industry']}/{most_recent['use_case']} was generated on {generated_at} with report date {report_date}."
                elif industry:
                    return f"The latest report for industry '{most_recent['industry']}' ({most_recent['use_case']}) was generated on {generated_at} with report date {report_date}."
                elif use_case:
                    return f"The latest report for use case '{most_recent['use_case']}' ({most_recent['industry']}) was generated on {generated_at} with report date {report_date}."
                else:
                    return f"The latest report in the system is for {most_recent['industry']}/{most_recent['use_case']}, generated on {generated_at} with report date {report_date}."
                    
            except Exception as e:
                logger.error(f"Error checking report dates: {e}")
                return f"Error checking report dates: {str(e)}"
        
        return check_latest_report_date
    
    def get_document_chunks(self, document_id: str, include_visual_refs: bool = True) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document using MongoDB connector.
        
        Args:
            document_id: Document identifier
            include_visual_refs: Whether to include visual references
            
        Returns:
            List of chunks sorted by index
        """
        return self.mongodb_connector.get_document_chunks(
            document_id=document_id,
            include_visual_refs=include_visual_refs
        )
    
    def get_chunks_with_visual_elements(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get chunks that contain visual elements using MongoDB connector.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of chunks with visual elements
        """
        return self.mongodb_connector.get_chunks_with_visual_elements(document_id)
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow for agentic RAG.
        
        Returns:
            Compiled LangGraph workflow
        """
        logger.info("🔧 Building Agentic RAG workflow")
        
        # Create workflow graph
        workflow = StateGraph(MessagesState)
        
        # Add nodes (using exact tutorial names)
        workflow.add_node("generate_query_or_respond", self.query_generator)
        workflow.add_node("retrieve", ToolNode(self.tools))
        workflow.add_node("rewrite_question", self.query_rewriter)
        workflow.add_node("generate_answer", self.answer_generator)
        
        # Add edges
        workflow.add_edge(START, "generate_query_or_respond")
        
        # Conditional routing from query generator (exact tutorial pattern)
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            }
        )
        
        # Conditional routing from retrieve (document grading)
        workflow.add_conditional_edges(
            "retrieve",
            self.document_grader,
            {
                "generate_answer": "generate_answer",
                "rewrite_question": "rewrite_question"
            }
        )
        
        # Final edges (exact tutorial pattern)
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        
        # Compile workflow with checkpointer for memory
        if self.checkpointer:
            compiled_workflow = workflow.compile(checkpointer=self.checkpointer)
            logger.info("✅ Agentic RAG workflow compiled with memory checkpointer")
        else:
            compiled_workflow = workflow.compile()
            logger.info("✅ Agentic RAG workflow compiled without memory")
        
        return compiled_workflow
    
    async def answer_with_agentic_rag(
        self,
        query: str,
        selected_documents: List[str] = None,
        max_chunks: int = 5,
        thread_id: Optional[str] = None,
        use_case: Optional[str] = None
    ) -> AgenticRAGResponse:
        """
        Answer a question using agentic RAG workflow.
        
        Args:
            query: User's question
            selected_documents: List of document IDs to use as context (None = all)
            max_chunks: Maximum number of chunks to retrieve
            thread_id: Thread ID for memory retrieval
            use_case: Use case for persona configuration
            
        Returns:
            AgenticRAGResponse with answer and workflow details
        """
        logger.info(f"🎯 Processing query with Agentic RAG: {query[:100]}...")
        
        # Set use case in query generator for persona lookup
        if use_case:
            self.query_generator.set_use_case(use_case)
            logger.info(f"🎭 Using persona for use case: {use_case}")
        
        # Clear previous citations at the start of each query
        self._last_citations = []
        
        # Store selected documents for the retriever tool to use
        self._selected_documents = selected_documents
        if selected_documents:
            logger.info(f"📋 Using selected documents for filtering: {selected_documents}")
        else:
            logger.info("📋 No document filtering - searching all documents")
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        # Track workflow steps
        workflow_steps = []
        grading_results = []
        query_rewrites = []
        
        try:
            # Execute workflow with memory using thread_id
            config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
            
            if thread_id:
                logger.info(f"🧠 Using memory with thread_id: {thread_id}")
            else:
                logger.info("🧠 No thread_id provided - conversation will not be persisted")
            
            # Execute workflow using astream() for async checkpointer
            final_state = initial_state
            async for chunk in self.workflow.astream(initial_state, config=config):
                for node_name, update in chunk.items():
                    workflow_steps.append(node_name)
                    logger.info(f"Executed node: {node_name}")
                    
                    # Update the state with the node's output (following tutorial pattern)
                    if "messages" in update:
                        final_state["messages"].extend(update["messages"])
                    
                    # Track specific node outputs
                    if node_name == "grade_documents":
                        # Extract grading result from the node
                        pass  # Will be stored in MongoDB by the node
                    elif node_name == "rewrite_question":
                        # Track query rewrites
                        if "messages" in update:
                            query_rewrites.append(update["messages"][-1].content)
            
            # Extract final answer from the final state messages
            final_messages = final_state.get("messages", [])
            logger.info(f"🔍 Final state has {len(final_messages)} messages")
            
            answer = "I apologize, but I couldn't generate a proper response."
            
            # Look for the last assistant message
            for i, message in enumerate(reversed(final_messages)):
                try:
                    logger.info(f"🔍 Message {i}: type={type(message)}, hasattr_content={hasattr(message, 'content')}")
                    
                    # Handle LangChain AIMessage objects
                    if hasattr(message, 'type') and message.type == "ai":
                        answer = message.content
                        logger.info(f"✅ Found AI message with content: {answer[:100]}...")
                        break
                    # Handle messages with role attribute
                    elif hasattr(message, 'role') and message.role == "assistant":
                        answer = message.content
                        logger.info(f"✅ Found assistant message with content: {answer[:100]}...")
                        break
                    # Handle dictionary messages
                    elif isinstance(message, dict) and message.get("role") == "assistant":
                        answer = message.get("content", answer)
                        logger.info(f"✅ Found dict assistant message with content: {answer[:100]}...")
                        break
                    # Handle any other message types that might have content
                    elif hasattr(message, 'content'):
                        answer = message.content
                        logger.info(f"✅ Found message with content attribute: {answer[:100]}...")
                        break
                except Exception as e:
                    logger.warning(f"⚠️ Error processing message {i}: {e}")
                    continue
            
            logger.info(f"📝 Extracted answer: {answer[:100]}...")
            
            # Get citations from the last retrieval
            citations = getattr(self, '_last_citations', [])
            
            # Create response
            response = AgenticRAGResponse(
                answer=answer,
                source_chunks=[],  # Will be populated from workflow
                source_documents=[],  # Will be populated from workflow
                confidence=0.8,  # Default confidence
                reasoning="Generated using agentic RAG workflow with document grading and query rewriting",
                citations=citations,  # Include source citations for modal
                workflow_steps=workflow_steps,
                grading_results=grading_results,
                query_rewrites=query_rewrites,
                session_id=thread_id
            )
            
            logger.info(f"✅ Agentic RAG completed with {len(workflow_steps)} steps")
            return response
            
        except Exception as e:
            logger.error(f"Error in agentic RAG workflow: {e}")
            return AgenticRAGResponse(
                answer=f"Error processing query: {str(e)}",
                source_chunks=[],
                source_documents=[],
                confidence=0.0,
                reasoning=f"Workflow error: {str(e)}",
                workflow_steps=workflow_steps
            )
