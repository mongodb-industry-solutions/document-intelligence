"""
Report Generation Service for Scheduled Reports
Generates PDF reports based on chunked documents using vector search and LLM
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from io import BytesIO
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environments

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from pydantic import BaseModel, Field

from langchain_aws import ChatBedrock
from db.mongodb_connector import MongoDBConnector
from tools.embedding_tools import generate_query_embedding_direct
from vogayeai.context_embeddings import VoyageContext3Embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeyMetric(BaseModel):
    """Model for a key metric extracted from documents"""
    label: str = Field(..., description="The metric name/label (max 50 chars)", max_length=50)
    value: str = Field(..., description="The actual value from the document (max 40 chars, SHORT)", max_length=40)
    trend: Optional[str] = Field(None, description="Optional trend - 1-2 words only (max 20 chars)", max_length=20)


class KeyMetricsResponse(BaseModel):
    """Response model for key metrics extraction"""
    metrics: List[KeyMetric] = Field(..., description="List of extracted key metrics")


class ReportGenerator:
    """
    Generates scheduled PDF reports for industry/use case combinations
    using vector search and LLM-powered content generation.
    """
    
    @staticmethod
    def format_use_case_title(use_case: str) -> str:
        """
        Format use case name for display, handling acronyms properly.
        
        Args:
            use_case: Use case code (e.g., 'kyc_onboarding', 'credit_rating')
            
        Returns:
            Properly formatted title (e.g., 'KYC Onboarding', 'Credit Rating')
        """
        # Define acronyms that should be uppercase
        acronyms = {'kyc', 'api', 'roi', 'kyb', 'aml', 'gdpr'}
        
        # Split by underscore and process each word
        words = use_case.split('_')
        formatted_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in acronyms:
                # Keep acronyms in uppercase
                formatted_words.append(word_lower.upper())
            else:
                # Title case for regular words
                formatted_words.append(word.capitalize())
        
        return ' '.join(formatted_words)
    
    def __init__(
        self,
        mongodb_connector: MongoDBConnector,
        embeddings_client: VoyageContext3Embeddings,
        llm: ChatBedrock = None,
        bedrock_client=None
    ):
        """
        Initialize Report Generator.
        
        Args:
            mongodb_connector: MongoDB connector for data access
            embeddings_client: VoyageAI embeddings client
            llm: Language model for content generation
            bedrock_client: AWS Bedrock client
        """
        self.mongodb_connector = mongodb_connector
        self.embeddings_client = embeddings_client
        
        # Initialize LLM if not provided
        if not llm:
            from cloud.aws.bedrock.client import BedrockClient
            if not bedrock_client:
                bedrock_client = BedrockClient()._get_bedrock_client()
            self.llm = ChatBedrock(
                model=os.getenv("BEDROCK_MODEL_ID"),
                client=bedrock_client,
                temperature=0.3,  # Lower temperature for consistent reports
                max_tokens=4096
            )
        else:
            self.llm = llm
            
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Create custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#001E2B'),
            spaceAfter=12,  # Reduced from 30 to minimize gap
            alignment=TA_CENTER
        ))
        
        # Section heading style
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#00684A'),
            spaceAfter=10,  # Reduced spacing after section heading
            spaceBefore=8   # Reduced spacing before section heading
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        ))
        
        # Disclaimer style
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            fontName='Helvetica-Oblique',
            alignment=TA_CENTER,
            spaceBefore=20,
            borderWidth=1,
            borderColor=colors.HexColor('#CCCCCC'),
            borderPadding=10,
            backColor=colors.HexColor('#F5F5F5')
        ))
        
        # Report date/time style
        self.styles.add(ParagraphStyle(
            name='ReportDateTime',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#666666'),
            fontName='Helvetica',
            alignment=TA_CENTER,
            spaceAfter=20,
            spaceBefore=5
        ))
    
    async def generate_report(
        self,
        industry: str,
        use_case: str,
        report_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Generate a scheduled report for an industry/use case combination.
        
        Args:
            industry: Industry code (e.g., 'fsi', 'healthcare')
            use_case: Use case code (e.g., 'credit_rating', 'risk_assessment')
            report_date: Date for the report (defaults to today)
            
        Returns:
            Dict with report metadata including file_path and statistics
        """
        try:
            if not report_date:
                report_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            
            logger.info(f"🎯 Generating report for {industry}/{use_case} on {report_date}")
            
            # 1. Get report template
            template = await self._get_report_template(industry, use_case)
            if not template:
                logger.warning(f"No template found for {industry}/{use_case}")
                return {"error": "No template found"}
            
            # 2. Gather initial relevant chunks using general vector search
            # This provides a fallback context if section-specific searches fail
            chunks = await self._gather_relevant_chunks(industry, use_case)
            if not chunks:
                logger.warning(f"No chunks found for {industry}/{use_case}")
                return {"error": "No data available"}
            
            # 3. Generate report content using LLM
            report_content = await self._generate_report_content(
                template, chunks, industry, use_case
            )
            
            # 4. Create PDF
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"{timestamp}_{industry}_{use_case}.pdf"
            file_path = f"/reports/{industry}/{filename}"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Generate PDF
            file_size = self._create_pdf(
                file_path, report_content, industry, use_case, report_date
            )
            
            # 5. Store report metadata in MongoDB
            report_metadata = {
                "industry": industry,
                "use_case": use_case,
                "report_date": report_date,
                "file_path": file_path,
                "file_size_kb": round(file_size / 1024, 2),  # Convert bytes to KB
                "total_pages": self._calculate_pages(report_content),
                "status": "generated",
                "generated_at": datetime.now(timezone.utc),
                "chunk_count": len(chunks),
                "document_count": len(set(chunk.get("document_id") for chunk in chunks))
            }
            
            # Save to MongoDB first
            self.mongodb_connector.scheduled_reports_collection.insert_one(report_metadata)
            
            # Clean up old reports AFTER successful generation (keep last 7)
            await self._cleanup_old_reports(industry, use_case)
            
            logger.info(f"✅ Report generated successfully: {file_path}")
            return report_metadata
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _get_report_template(self, industry: str, use_case: str) -> Optional[Dict[str, Any]]:
        """Get report template from MongoDB."""
        template = self.mongodb_connector.report_templates_collection.find_one({
            "industry": industry,
            "use_case": use_case,
            "is_active": True
        })
        return template
    
    async def _gather_section_chunks(
        self,
        industry: str,
        use_case: str,
        section: Dict[str, Any],
        previous_context: List[Dict[str, Any]],
        max_chunks: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Gather section-specific chunks using targeted vector search.
        
        Args:
            industry: Industry code
            use_case: Use case code
            section: Section configuration from template
            previous_context: Previous section summaries for context
            max_chunks: Maximum number of chunks to retrieve
            
        Returns:
            List of relevant chunks for this specific section
        """
        try:
            # Create section-specific semantic query
            section_title = section['title']
            section_prompt = section['prompt']
            
            # Build a more targeted query for this section
            query_parts = [
                f"Find information about {section_title} for {use_case.replace('_', ' ')} in {industry}.",
                section_prompt
            ]
            
            # Add context from previous sections if available
            if previous_context:
                recent_context = previous_context[-1]['summary'] if previous_context else ""
                query_parts.append(f"Building upon: {recent_context}")
            
            section_query = " ".join(query_parts)
            
            # Log the section-specific query
            logger.info(f"🔍 Section-specific search for '{section_title}': {section_query[:100]}...")
            
            # Generate embedding
            embedding_result = generate_query_embedding_direct(
                query=section_query,
                embeddings_client=self.embeddings_client
            )
            
            if not embedding_result["success"]:
                logger.error(f"Failed to generate query embedding for section {section_title}")
                return []
            
            # Perform vector search with filter
            filter_query = {
                "$or": [
                    {"metadata.path": {"$regex": f".*/{industry}/{use_case}/.*"}},
                    {"metadata.path": {"$regex": f".*/{industry}/.*{use_case}.*"}}
                ]
            }
            
            chunks = self.mongodb_connector.vector_search(
                query_embedding=embedding_result["query_embedding"],
                k=max_chunks,
                filter=filter_query
            )
            
            logger.info(f"Found {len(chunks)} chunks for section '{section_title}'")
            return chunks
            
        except Exception as e:
            logger.error(f"Error gathering section chunks: {e}")
            return []
    
    async def _gather_relevant_chunks(
        self,
        industry: str,
        use_case: str,
        max_chunks: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Gather initial relevant chunks using a general vector search.
        This provides a broad context and fallback for section-specific searches.
        
        Args:
            industry: Industry code
            use_case: Use case code
            max_chunks: Maximum number of chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        try:
            # Create semantic query based on industry and use case
            query = f"Generate a comprehensive {use_case.replace('_', ' ')} report for the {industry} industry"
            
            # Generate embedding
            embedding_result = generate_query_embedding_direct(
                query=query,
                embeddings_client=self.embeddings_client
            )
            
            if not embedding_result["success"]:
                logger.error("Failed to generate query embedding")
                return []
            
            # Perform vector search with filter
            # Filter by document paths that match the industry/use case pattern
            filter_query = {
                "$or": [
                    {"metadata.path": {"$regex": f".*/{industry}/{use_case}/.*"}},
                    {"metadata.path": {"$regex": f".*/{industry}/.*{use_case}.*"}}
                ]
            }
            
            chunks = self.mongodb_connector.vector_search(
                query_embedding=embedding_result["query_embedding"],
                k=max_chunks,
                filter=filter_query
            )
            
            logger.info(f"Found {len(chunks)} relevant chunks for {industry}/{use_case}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error gathering chunks: {e}")
            return []
    
    async def _extract_key_metrics(
        self,
        chunks: List[Dict[str, Any]],
        key_metrics_config: Dict[str, Any],
        use_case: str,
        industry: str
    ) -> List[Dict[str, str]]:
        """
        Extract key metrics from document chunks using LLM with structured output.
        
        Args:
            chunks: Document chunks to analyze
            key_metrics_config: Configuration from template
            use_case: Use case code
            industry: Industry code
            
        Returns:
            List of key metrics with label, value, and optional trend
        """
        try:
            logger.info(f"🔢 Extracting key metrics for {industry}/{use_case}")
            
            # Get configuration
            metrics_count = key_metrics_config.get("count", 4)
            guidance_prompt = key_metrics_config.get("guidance_prompt", "")
            
            # Prepare context from top chunks
            context = "\n\n".join([
                f"[Document: {chunk.get('document_name', 'Unknown')}]\n{chunk.get('chunk_text', '')}"
                for chunk in chunks[:20]  # Use top 20 chunks for metrics
            ])
            
            # Create extraction prompt
            prompt = f"""
            You are analyzing documents for a {use_case.replace('_', ' ')} report in the {industry} industry.
            
            Your task: Identify the {metrics_count} most important metrics, scores, ratings, or key statistics from these documents.
            
            Guidance: {guidance_prompt}
            
            Document Content:
            {context}
            
            CRITICAL INSTRUCTIONS:
            - Extract ACTUAL values that appear in the documents - do NOT invent or estimate
            - Keep the VALUE field SHORT and CONCISE (maximum 30 characters)
            - Value should be just the number/rating/metric - NO explanations
            - Trend field should be SHORT (1-2 words max): "Stable", "Improved", "Declined", "Strong", etc.
            - Do NOT include long explanations or context in the value or trend fields
            - Return EXACTLY {metrics_count} metrics
            
            Examples of CORRECT format:
            - Label: "Credit Rating", Value: "BBB+", Trend: "Positive"
            - Label: "Debt/EBITDA", Value: "2.1x", Trend: "Improved"
            - Label: "Disputed Amount", Value: "$2,450.00", Trend: ""
            - Label: "Resolution Time", Value: "45 days", Trend: "On-time"
            
            Examples of WRONG format (too verbose):
            - Label: "Credit Rating", Value: "BBB+ with positive outlook considering...", Trend: "Improved from previous..."
            - Label: "Debt/EBITDA", Value: "2.1x which is better than the projected 2.3x...", Trend: "..."
            
            Keep it SHORT and SIMPLE. Return your response as a JSON object with a "metrics" array.
            """
            
            # Use structured output for reliable JSON parsing
            structured_llm = self.llm.with_structured_output(KeyMetricsResponse)
            response = structured_llm.invoke(prompt)
            
            # Convert to list of dicts with length constraints
            metrics_list = []
            for metric in response.metrics[:metrics_count]:  # Ensure we have exactly the count requested
                # Truncate label if too long (max 50 chars)
                label = metric.label[:50] if len(metric.label) > 50 else metric.label
                
                # Truncate value if too long (max 40 chars)
                value = metric.value[:40] if len(metric.value) > 40 else metric.value
                
                # Truncate trend if too long (max 20 chars) and keep it short
                trend = ""
                if metric.trend:
                    trend = metric.trend[:20] if len(metric.trend) > 20 else metric.trend
                
                metric_dict = {
                    "label": label,
                    "value": value
                }
                if trend:
                    metric_dict["trend"] = trend
                metrics_list.append(metric_dict)
            
            # Pad with "N/A" metrics if we got fewer than requested
            while len(metrics_list) < metrics_count:
                metrics_list.append({
                    "label": f"Metric {len(metrics_list) + 1}",
                    "value": "Not Available",
                    "trend": None
                })
            
            logger.info(f"✅ Extracted {len(metrics_list)} key metrics")
            return metrics_list
            
        except Exception as e:
            logger.error(f"Error extracting key metrics: {e}")
            # Return empty list on error - report will continue without metrics
            return []
    
    async def _generate_report_content(
        self,
        template: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        industry: str,
        use_case: str
    ) -> Dict[str, Any]:
        """
        Generate report content using LLM based on template and chunks.
        Uses section-specific vector search for more accurate context retrieval.
        
        Args:
            template: Report template with sections
            chunks: Initial relevant document chunks (used as fallback)
            industry: Industry code
            use_case: Use case code
            
        Returns:
            Dict with generated content for each section
        """
        report_content = {}
        accumulated_context = []  # Store generated content for context consistency
        
        # Check if key metrics are enabled in template
        key_metrics_config = template["structure"].get("key_metrics")
        if key_metrics_config and key_metrics_config.get("enabled", False):
            # Extract key metrics FIRST (before sections)
            logger.info("🔢 Key metrics enabled - extracting metrics from documents")
            metrics = await self._extract_key_metrics(
                chunks=chunks,
                key_metrics_config=key_metrics_config,
                use_case=use_case,
                industry=industry
            )
            if metrics:
                # Store with underscore prefix to distinguish from regular sections
                report_content["_key_metrics"] = metrics
                logger.info(f"✅ Stored {len(metrics)} key metrics for report")
        
        # Generate content for each section
        for section in template["structure"]["sections"]:
            section_key = section["title"].lower().replace(" ", "_")
            
            # Perform section-specific vector search
            section_chunks = await self._gather_section_chunks(
                industry=industry,
                use_case=use_case,
                section=section,
                previous_context=accumulated_context
            )
            
            # Use section-specific chunks if found, otherwise fall back to general chunks
            relevant_chunks = section_chunks if section_chunks else chunks
            
            # Prepare context from chunks
            context = "\n\n".join([
                f"[Document: {chunk.get('document_name', 'Unknown')}]\n{chunk.get('chunk_text', '')}"
                for chunk in relevant_chunks[:30]  # Limit context to avoid token limits
            ])
            
            # Add previous section summaries for consistency (if any)
            if accumulated_context:
                previous_sections_summary = "\n\n".join([
                    f"[Previous Section - {ctx['title']}]: {ctx['summary']}"
                    for ctx in accumulated_context[-2:]  # Last 2 sections for context
                ])
                context = f"{previous_sections_summary}\n\n{context}"
            
            # Simplified prompt for all content types
            if section['content_type'] == 'table':
                # Convert table requests to structured text
                prompt = f"""
                Generate structured content for the section: {section['title']}
                
                Task: {section['prompt']}
                
                CRITICAL INSTRUCTION: Do NOT repeat the section title "{section['title']}" in your response.
                Start IMMEDIATELY with the substantive content. No title, no heading, no introduction.
                
                Document Context:
                {context}
                
                Requirements:
                - Maximum {section['max_words']} words
                - Format as structured text with clear sections
                - Use bullet points or numbered lists for data
                - Include specific metrics and values from the documents
                - Format as: "Metric: Value (Benchmark: X)" or "• Key Point: Details"
                - Be professional and concise
                {"- Ensure consistency with previous sections' findings" if accumulated_context else ""}
                
                Generate the structured content (WITHOUT repeating the section title):
                """
            else:
                prompt = f"""
                Generate professional content for the section: {section['title']}
                
                Task: {section['prompt']}
                
                CRITICAL INSTRUCTION: Do NOT repeat the section title "{section['title']}" in your response.
                Start IMMEDIATELY with the substantive content. No title, no heading, no introduction.
                
                Document Context:
                {context}
                
                Requirements:
                - Maximum {section['max_words']} words
                - Write in professional, executive report style
                - Use specific data and metrics from the documents
                - Avoid repetitive phrases or unnecessary clarifications
                - Start directly with the content, no introductory phrases like "Here's a..." or "Based on the documents..."
                - Be concise and actionable
                {"- Ensure consistency with previous sections' findings" if accumulated_context else ""}
                
                Generate the content (WITHOUT repeating the section title):
                """
            
            response = self.llm.invoke(prompt)
            
            # Clean up the content to remove repetitive phrases and title duplication
            cleaned_content = self._clean_content(response.content, section_title=section["title"])
            
            report_content[section_key] = {
                "title": section["title"],
                "content": cleaned_content,
                "content_type": section["content_type"],
                "order": section["order"]
            }
            
            # Add to accumulated context for next sections
            # Create a brief summary for context (first 150 chars)
            summary = cleaned_content[:150] + "..." if len(cleaned_content) > 150 else cleaned_content
            accumulated_context.append({
                "title": section["title"],
                "summary": summary
            })
        
        return report_content
    
    def _clean_content(self, content: str, section_title: str = None) -> str:
        """
        Clean up content to remove repetitive phrases and improve readability.
        
        Args:
            content: Raw content from LLM
            section_title: Section title to remove if found at start (optional)
            
        Returns:
            Cleaned content
        """
        # First, remove section title if it appears at the beginning
        if section_title:
            content = self._remove_title_repetition(content, section_title)
        
        # Remove common repetitive phrases
        phrases_to_remove = [
            "Here's a ",
            "Based on the documents",
            "Based on the provided documents",
            "Here is a ",
            "The following ",
            "This section ",
            "In this section ",
            "The document analysis ",
            "The analysis shows ",
            "The data indicates ",
            "The information reveals ",
            "The findings suggest ",
            "The results demonstrate ",
            "The evidence shows ",
            "The report indicates ",
            "The assessment reveals ",
            "The evaluation shows ",
            "The review indicates ",
            "The study reveals ",
            "The investigation shows ",
            "The examination indicates ",
            "The analysis reveals ",
            "The research shows ",
            "The findings indicate ",
            "The results show ",
            "The data shows ",
            "The information shows ",
            "The evidence indicates ",
            "The report shows ",
            "The assessment shows ",
            "The evaluation indicates ",
            "The review shows ",
            "The study indicates ",
            "The investigation indicates ",
            "The examination shows ",
            "The analysis shows ",
            "The research indicates ",
        ]
        
        cleaned = content
        for phrase in phrases_to_remove:
            # Remove phrase at the beginning of sentences
            cleaned = cleaned.replace(phrase, "")
            # Remove phrase at the beginning of paragraphs
            cleaned = cleaned.replace(f"\n{phrase}", "\n")
            # Remove phrase at the beginning of content
            if cleaned.startswith(phrase):
                cleaned = cleaned[len(phrase):].strip()
        
        # Clean up whitespace while preserving paragraph structure
        lines = cleaned.split("\n")
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Keep non-empty lines
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1]:  # Add single empty line between paragraphs
                cleaned_lines.append("")
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
            
        return "\n".join(cleaned_lines)
    
    def _remove_title_repetition(self, content: str, title: str) -> str:
        """
        Remove section title if it appears at the beginning of content.
        Handles various formatting patterns that LLMs might use.
        
        Args:
            content: Content to clean
            title: Section title to look for
            
        Returns:
            Content with title removed if found at the beginning
        """
        # Strip leading whitespace for matching
        content_stripped = content.lstrip()
        
        # Patterns to check (in order of specificity):
        # - Markdown headers: "## Title", "# Title"
        # - Bold markdown: "**Title**"
        # - With punctuation: "Title:", "Title -", "Title –"
        # - Plain with newline: "Title\n", "Title\r\n"
        # - Plain at start (if followed by punctuation or whitespace)
        patterns = [
            f"## {title}",
            f"# {title}",
            f"**{title}**",
            f"{title}:",
            f"{title} -",
            f"{title} –",  # en dash
            f"{title} —",  # em dash
            f"{title}\n",
            f"{title}\r\n",
        ]
        
        # Try to match each pattern at the start
        for pattern in patterns:
            if content_stripped.startswith(pattern):
                # Remove the pattern and any following whitespace/newlines
                remaining = content_stripped[len(pattern):].lstrip()
                logger.debug(f"Removed title pattern '{pattern}' from content start")
                return remaining
        
        # Check if title appears at start without specific punctuation
        # but followed by a newline within the first few characters
        if content_stripped.startswith(title):
            after_title = content_stripped[len(title):]
            # If immediately followed by newline(s) or end of string, remove it
            if after_title.startswith('\n') or after_title.startswith('\r') or after_title == '':
                remaining = after_title.lstrip()
                logger.debug(f"Removed plain title '{title}' from content start")
                return remaining
        
        # No title repetition found, return original content
        return content
    
    def _create_key_metrics_table(self, metrics: List[Dict[str, str]]) -> Table:
        """
        Create a simple vertical list of key metrics boxes.
        Much simpler than complex table - easier to read and no overlap issues.
        
        Args:
            metrics: List of metrics with label, value, and optional trend
            
        Returns:
            ReportLab Table object
        """
        # Create a simple 2-column table: Metric | Value
        table_data = []
        for metric in metrics:
            label = metric["label"]
            value = metric["value"]
            trend = metric.get("trend", "")
            
            # Combine value and trend if trend exists - keep it short
            if trend:
                full_value = f"{value} ({trend})"
            else:
                full_value = value
            
            # Wrap text in Paragraphs for proper line breaking
            label_para = Paragraph(label, self.styles['Normal'])
            value_para = Paragraph(full_value, self.styles['Normal'])
            
            table_data.append([label_para, value_para])
        
        # Create simple 2-column table with wider columns
        metrics_table = Table(
            table_data,
            colWidths=[2.2 * inch, 4.0 * inch]  # Wider value column for wrapping
        )
        
        # Simple, clean styling
        metrics_table.setStyle(TableStyle([
            # All cells
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),  # Labels column bold
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#00684A')),  # Labels in green
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),  # Values in black
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            
            # Borders
            ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#00684A')),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor('#E0E0E0')),  # Lines between rows
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F8F8')),  # Light background
        ]))
        
        return metrics_table
    
    def _create_pdf(
        self,
        file_path: str,
        report_content: Dict[str, Any],
        industry: str,
        use_case: str,
        report_date: datetime
    ) -> int:
        """
        Create PDF file from report content.
        
        Args:
            file_path: Path to save PDF
            report_content: Generated content for each section
            industry: Industry code
            use_case: Use case code
            report_date: Report date
            
        Returns:
            File size in bytes
        """
        # Create PDF document
        doc = SimpleDocTemplate(
            file_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Add title
        title_text = f"{self.format_use_case_title(use_case)} Report"
        elements.append(Paragraph(title_text, self.styles['CustomTitle']))
        
        # Add report generation date and time
        # Get current UTC time for generation timestamp
        generation_time = datetime.now(timezone.utc)
        date_time_text = f"Generated on {generation_time.strftime('%B %d, %Y at %I:%M %p UTC')}"
        elements.append(Paragraph(date_time_text, self.styles['ReportDateTime']))
        
        elements.append(Spacer(1, 0.3*inch))  # Proper spacing after date/time
        
        # Sort sections by order (exclude _key_metrics which starts with underscore)
        sorted_sections = sorted(
            [(k, v) for k, v in report_content.items() if not k.startswith('_')],
            key=lambda x: x[1].get('order', 999)
        )
        
        # Add each section
        first_section_added = False
        for section_key, section_data in sorted_sections:
            # Special handling for disclaimer section
            if section_data['title'].lower() == 'disclaimer':
                # Add some space before disclaimer
                elements.append(Spacer(1, 0.5*inch))
                elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
                elements.append(Spacer(1, 0.2*inch))
                
                # Add disclaimer with special formatting
                disclaimer_content = section_data['content'].strip()
                elements.append(Paragraph(disclaimer_content, self.styles['Disclaimer']))
            else:
                # Section heading
                elements.append(Paragraph(section_data['title'], self.styles['SectionHeading']))
                
                # For FIRST section: Add key metrics table BEFORE content
                if not first_section_added and "_key_metrics" in report_content:
                    logger.info("📊 Adding key metrics table after first section heading")
                    elements.append(Spacer(1, 0.15*inch))  # Small space after heading
                    metrics_table = self._create_key_metrics_table(report_content["_key_metrics"])
                    elements.append(metrics_table)
                    elements.append(Spacer(1, 0.25*inch))  # Space after metrics before content
                    first_section_added = True
                
                # Simplified content handling - all content as text
                content = section_data['content']
                
                # Split content into paragraphs, handling both \n\n and single \n
                paragraphs = content.split('\n\n')
                
                # If no double newlines, split by single newlines and group
                if len(paragraphs) == 1 and '\n' in content:
                    lines = content.split('\n')
                    paragraphs = []
                    current_para = []
                    
                    for line in lines:
                        line = line.strip()
                        if line:
                            current_para.append(line)
                        elif current_para:
                            paragraphs.append(' '.join(current_para))
                            current_para = []
                    
                    if current_para:
                        paragraphs.append(' '.join(current_para))
                
                # Add each paragraph
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        elements.append(Paragraph(para, self.styles['CustomBody']))
            
            # Normal spacing between sections
            elements.append(Spacer(1, 0.15*inch))
        
        # Add footer
        elements.append(Spacer(1, 0.5*inch))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        elements.append(Paragraph(
            "This report was automatically generated by MongoDB Document Intelligence System",
            self.styles['Footer']
        ))
        
        # Build PDF
        doc.build(elements)
        
        # Return file size
        return os.path.getsize(file_path)
    
    
    def _create_sample_chart(self, section_key: str) -> Optional[Image]:
        """
        Create a sample chart for the report.
        
        Args:
            section_key: Section identifier
            
        Returns:
            Image object or None
        """
        try:
            # Create a simple bar chart
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Sample data
            categories = ['Q1', 'Q2', 'Q3', 'Q4']
            values = [75, 82, 78, 85]
            
            # Create bar chart
            bars = ax.bar(categories, values, color='#00684A')
            
            # Customize chart
            ax.set_ylabel('Performance Score')
            ax.set_title(f'{section_key.replace("_", " ").title()} Metrics')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height}%', ha='center', va='bottom')
            
            # Save to BytesIO
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            plt.close()
            
            # Create ReportLab Image
            img = Image(img_buffer, width=4*inch, height=2.5*inch)
            return img
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return None
    
    def _calculate_pages(self, report_content: Dict[str, Any]) -> int:
        """
        Calculate more accurate page count based on content structure.
        
        Args:
            report_content: Generated report content
            
        Returns:
            Estimated page count
        """
        try:
            # Count sections and their content
            section_count = len(report_content)
            total_chars = 0
            
            for section in report_content.values():
                content = section.get('content', '')
                total_chars += len(content)
            
            # Better estimation based on:
            # - Each section title takes ~1 line
            # - Each section has spacing
            # - Average ~2000 characters per page (including formatting)
            # - Minimum 1 page, maximum 4 pages
            estimated_pages = max(1, min(4, (total_chars // 2000) + section_count // 3))
            
            return estimated_pages
            
        except Exception as e:
            logger.warning(f"Error calculating pages: {e}")
            return 3  # Default fallback
    
    async def _cleanup_old_reports(self, industry: str, use_case: str):
        """
        Clean up old reports, keeping only the last 7.
        ALWAYS keeps at least 1 report as fallback.
        
        Args:
            industry: Industry code
            use_case: Use case code
        """
        try:
            # Find all reports for this industry/use case
            reports = list(self.mongodb_connector.scheduled_reports_collection.find({
                "industry": industry,
                "use_case": use_case
            }).sort("generated_at", -1))
            
            # Keep only the last 7, but ALWAYS keep at least 1
            if len(reports) > 7:
                reports_to_delete = reports[7:]
                
                for report in reports_to_delete:
                    try:
                        # Delete file
                        file_path = report.get("file_path")
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"Deleted old report file: {file_path}")
                        
                        # Delete from MongoDB
                        self.mongodb_connector.scheduled_reports_collection.delete_one({
                            "_id": report["_id"]
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error deleting report {report.get('_id')}: {e}")
                        # Continue with other deletions even if one fails
                        continue
                    
                logger.info(f"Cleaned up {len(reports_to_delete)} old reports for {industry}/{use_case}")
            else:
                logger.info(f"Keeping all {len(reports)} reports for {industry}/{use_case} (within limit)")
            
        except Exception as e:
            logger.error(f"Error cleaning up old reports: {e}")
            # Don't raise exception - cleanup failure shouldn't break report generation
    
    async def get_latest_report(
        self,
        industry: str,
        use_case: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest report for an industry/use case combination.
        
        Args:
            industry: Industry code
            use_case: Use case code
            
        Returns:
            Latest report metadata or None
        """
        report = self.mongodb_connector.scheduled_reports_collection.find_one({
            "industry": industry,
            "use_case": use_case,
            "status": "generated"
        }, sort=[("generated_at", -1)])
        
        return report
