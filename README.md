# FSI Document Intelligence Demo

A MongoDB Atlas-powered intelligent document processing system that demonstrates how a unified data platform enables sophisticated AI workflows. This implementation showcases **three distinct architectural patterns**, all built on MongoDB as the backbone:

1. **Supervisor Multi-Agent Orchestration** - MongoDB stores workflow states, document metadata, and assessments across all agents
2. **Agentic RAG System** - MongoDB's vector search and checkpointing enable self-correcting Q&A with conversation memory
3. **Automated Report Generation** - MongoDB's flexible schema stores templates, schedules, and enables semantic search across sections

**Why MongoDB is Central**: Unlike traditional architectures that require separate databases for structured data, documents, vectors, and operational state, MongoDB Atlas serves as the single source of truth for all data types. This unified approach eliminates data silos, reduces complexity, and enables real-time AI decision-making.

**Key Technologies**: The system leverages MongoDB Atlas as its foundation, enhanced with LangGraph for orchestration, VoyageAI's voyage-context-3 for embeddings, and AWS Bedrock's Claude 3.5 Sonnet v2 for vision AI.

## 🚀 Quick Start

1. **Prerequisites**:
   - **Required**: MongoDB Atlas account with connection string
   - **Required**: VoyageAI API key for embeddings
   - **Required**: AWS account with Bedrock access (Claude 3.5 Sonnet v2)
   - **Optional**: AWS SSO configured for S3 document access

2. **Environment Setup** (⚠️ CRITICAL - Demo won't work without this):
   ```bash
   # Create the backend .env file
   cd backend
   cp .env.example .env  # If example exists
   # Edit .env with your actual values (see Configuration & Environment Variables section below)
   ```

3. **Database Setup**:
   ```bash
   # Import seed data (from project root)
   cd backend/db/collections/seeds
   
   # OPTIONAL: Only if using S3/GDrive sources (not required for local file upload)
   # Copy and configure S3/GDrive settings if needed
   cp document_intelligence.buckets.example.json document_intelligence.buckets.json
   cp document_intelligence.gdrive.example.json document_intelligence.gdrive.json
   # Edit these files with your actual bucket/folder IDs
   
   # Import essential collections
   mongoimport --uri "$MONGODB_URI" --collection industry_mappings --file document_intelligence.industry_mappings.json --jsonArray
   mongoimport --uri "$MONGODB_URI" --collection report_templates --file document_intelligence.report_templates.json --jsonArray
   mongoimport --uri "$MONGODB_URI" --collection agent_personas --file document_intelligence.agent_personas.json --jsonArray
   
   # Import S3/GDrive configs only if using those sources
   mongoimport --uri "$MONGODB_URI" --collection buckets --file document_intelligence.buckets.json --jsonArray
   mongoimport --uri "$MONGODB_URI" --collection gdrive --file document_intelligence.gdrive.json --jsonArray
   
   # Optional: Import other sample data for testing
   # See all available seed files in backend/db/collections/seeds/
   ```
   
   > **💡 Quick Demo**: For a fast start with local files only:
   > - Skip S3/GDrive configuration (steps above marked OPTIONAL)
   > - Use sample documents in `backend/data/seed/docs/`
   > - View pre-generated reports in `backend/data/seed/reports/`
   > - Upload your own documents via the API at `http://localhost:8080/api/upload/documents`

4. **Run the Application**:
   ```bash
   # From project root
   docker-compose up --build
   ```

5. **Create Vector Search Index** (Required for Q&A functionality):
   - Go to MongoDB Atlas → your cluster → Atlas Search
   - Create new index on `chunks` collection
   - Index name: `document_intelligence_chunks_vector_index`
   - Use this JSON configuration:
   ```json
   {
     "fields": [{
       "type": "vector",
       "path": "embedding",
       "numDimensions": 1024,
       "similarity": "cosine"
     }]
   }
   ```

## 🏗️ System Architecture

### High-Level Architecture

![High Level Architecture](diagrams/1_high_level_architecture.png)

**Key Features:**
- 🎯 **Smart Ingestion**: Context-aware assessment based on industry and topic
- 👁️ **Pure Vision Understanding**: Claude 3.5 Sonnet v2 for document extraction
- 🔍 **Context-Aware Embeddings**: Each chunk maintains full document context via voyage-context-3
- 🤖 **Multi-Agent Architecture**: Specialized agents with LangGraph orchestration
- 🗄️ **Multi-Source Support**: Local files, AWS S3, and Google Drive with unified workflow
- 🔒 **Deduplication**: Intelligent caching prevents reprocessing
- 🏭 **Industry-Specific**: Configurable mappings for different verticals
- 🔄 **Agentic RAG**: Self-correcting Q&A with document grading and query rewriting
- 📊 **Scheduled Reports**: Automated PDF generation with section-specific semantic search


## 🏛️ Architectural Patterns

### Part 1: Supervisor Multi-Agent Orchestration

![Supervisor Multi-Agent Orchestration](diagrams/2_part1_ingestion_multiagent_supervisor.png)

### Ingestion Workflow

```mermaid
graph TD
    A[Document Sources] --> B[🧠 Supervisor Agent<br/>Central Coordinator]
    A1[📁 Local Files] --> A
    A2[☁️ AWS S3] --> A
    A3[📊 Google Drive] --> A
    
    B --> C{Routing Decision}
    C -->|"1. Discover"| D[📂 Scanner Agent<br/>Document Discovery]
    C -->|"2. Assess"| E[🔍 Evaluator Agent<br/>Relevance Check]
    C -->|"3. Extract"| F[📸 Extractor Agent<br/>Vision AI]
    C -->|"4. Process"| G[💾 Processor Agent<br/>Chunk & Embed]
    C -->|Complete| H[✅ End]
    
    D -->|Found Docs| B
    E -->|Assessment| B
    F -->|Markdown| B
    G -->|Stored| B
    
    subgraph "MongoDB Atlas Storage"
        P1[documents]
        P2[chunks]
        P3[assessments]
        P4[workflows]
    end
    
    G --> P1
    G --> P2
    E --> P3
    B --> P4
    
    style B fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    style D fill:#f3e5f5
    style E fill:#e8f5e8
    style F fill:#fff3e0
    style G fill:#fce4ec
```

**Pattern**: The **supervisor pattern** is a multi-agent architecture where a central supervisor agent coordinates specialized worker agents. This approach excels when tasks require different types of expertise. Rather than building one agent that manages tool selection across domains, we create focused specialists coordinated by a supervisor who understands the overall workflow.

![Supervisor Pattern Explanation](diagrams/3_part1_multiagent_supervisor_pattern_explanation.png)

In our Document Intelligence system, the supervisor agent orchestrates:
- **Scanner Agent**: Discovers documents from multiple sources (Local files, AWS S3, Google Drive)
- **Evaluator Agent**: Assesses document relevance based on industry and use case context
- **Extractor Agent**: Extracts content using vision AI (Claude 3.5 Sonnet v2) and converts to markdown
- **Processor Agent**: Splits documents into chunks and generates context-aware embeddings (voyage-context-3)

**Why Use a Supervisor?**

Following the [LangChain supervisor pattern](https://docs.langchain.com/oss/python/langchain/supervisor), our multi-agent architecture allows us to:
- **Partition tools across workers**: Each agent has access only to relevant tools (e.g., Scanner has S3/GDrive access, Extractor has vision AI)
- **Focused expertise**: Each agent has individual prompts and instructions specific to their domain
- **Manage complexity**: Instead of one agent handling document discovery, evaluation, extraction, AND processing, we separate concerns
- **Iterative improvement**: If performance degrades, we can improve individual agents without affecting the entire system

**MongoDB Value for Ingestion**:
- **Unified Data Platform**: Structured metadata, unstructured documents, and vector embeddings all in one database
- **Workflow Tracking**: Real-time state persistence across all agents
- **Document Deduplication**: Prevent reprocessing with document status tracking
- **Assessment**: Industry-specific relevance scores and decisions

### Part 2: Agentic RAG System

![Agentic RAG Architecture](diagrams/4_part2_QandA_agentic_rag.png)

### Agentic RAG Flow

```mermaid
graph TD
    A[User Query] --> B[🤖 Generate Query or Respond<br/>Retrieval Agent]
    B --> C{Tool Call?}
    
    C -->|No Tool Call| D[Direct Response]
    C -->|retrieve_documents| E[🔧 Retrieve Tool<br/>Semantic Search]
    
    E --> F[MongoDB Atlas<br/>Vector Search]
    F --> G[📄 Retrieved Chunks]
    G --> H[🔍 Grade Documents<br/>Relevance Check]
    
    H --> I{Relevant?}
    I -->|Yes| J[📝 Generate Answer]
    I -->|No| K[✏️ Rewrite Question]
    
    K --> L[Improved Query]
    L --> B
    
    J --> M[Answer with Citations]
    D --> N[Response to User]
    M --> N
    
    subgraph "MongoDB Collections"
        O[chunks]
        P[gradings]
        Q[checkpoints_aio]
        R[agent_personas]
    end
    
    F --> O
    H --> P
    B --> Q
    B --> R
    
    style B fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style H fill:#e8f5e8
    style K fill:#fff3e0
    style J fill:#fce4ec
    style E fill:#d4e157
```

**Pattern**: Agentic Retrieval-Augmented Generation with Self-Correction

Following the [LangGraph Agentic RAG pattern](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/), our Q&A system implements a retrieval agent that makes intelligent decisions about whether to retrieve context from MongoDB Atlas Vector Search or respond directly to the user.

![Agentic RAG Pattern Explanation](diagrams/5_part2_agentic_rag_pattern_explanation.png)

**Key Components**:

1. **Retrieval Agent (Query Generator)**: 
   - Decides whether to retrieve context using semantic search or respond directly
   - Uses voyage-context-3 embeddings for semantic search
   - Bound with retriever tool that searches MongoDB chunks collection

2. **Document Grader (Conditional Edge)**:
   - Grades retrieved documents for relevance to the user question
   - Returns binary score ('yes' or 'no') for each chunk
   - Routes to answer generation if relevant, query rewriting if not

3. **Query Rewriter**:
   - Self-correcting mechanism when retrieved documents aren't relevant
   - Reformulates the question for better retrieval results
   - Loops back to the retrieval agent for another attempt

4. **Answer Generator**:
   - Synthesizes final answer from relevant retrieved chunks
   - Maintains citations and source tracking
   - Provides comprehensive answers based on context

**MongoDB Integration**:
- **Semantic Search**: Find relevant content based on meaning, not just keywords
- **Memory Persistence**: checkpoint_writes_aio and checkpoints_aio for conversation state
- **Agent Personas**: Use-case specific configurations (Credit Rating Analyst, Investment Research Analyst, etc.)
- **Grading Storage**: Stores document relevance assessments for analysis

**Why Agentic RAG?**

Unlike traditional RAG that always retrieves, our system:
- **Makes intelligent decisions**: Knows when retrieval is necessary vs. direct response
- **Self-corrects**: If retrieved documents aren't relevant, rewrites query and tries again
- **Maintains context**: Uses MongoDB checkpointing for multi-turn conversations
- **Provides transparency**: Tracks workflow steps and grading decisions

**Workflow Steps**:

1. **User Query** → **Generate Query or Respond**
   - LLM decides: answer directly or search for information
   - Has access to `retrieve_documents` tool via `.bind_tools()`
   - For greetings/simple questions → direct response
   - For document-specific questions → tool call

2. **Tool Call Decision** → **Retrieve Tool**
   - If tool called: Generate voyage-context-3 embedding
   - Search MongoDB chunks collection with optional document filtering
   - Return top-k relevant chunks with metadata

3. **Grade Documents** → **Routing Decision**
   - Each retrieved chunk is graded for relevance
   - Binary score: 'yes' (relevant) or 'no' (not relevant)
   - Grading results stored in MongoDB for analysis

4. **Answer Generation or Query Rewriting**
   - If relevant → Generate comprehensive answer with citations
   - If not relevant → Rewrite query and loop back to step 1
   - Maximum iterations prevent infinite loops

**Implementation Details**:
- Uses `MessagesState` for state management
- `tools_condition` for conditional routing after tool calls
- Custom grading prompt for document relevance assessment
- Structured output (`GradeDocuments` schema) for consistent grading
- Thread-based memory with MongoDB checkpointer for conversation persistence

**MongoDB Value for Q&A**:
- **Unified Data Platform**: Structured metadata, unstructured text chunks, and vector embeddings coexist seamlessly
- **Vector Search**: Lightning-fast semantic search across millions of documents
- **Conversation Memory**: Checkpointing system for multi-turn dialogue persistence
- **Agent Personas**: Store use-case specific configurations and prompts

### Part 3: Scheduled Reports

![Scheduled Reports Architecture](diagrams/6_part3_scheduled_reports.png)

**Pattern**: Automated Report Generation with Section-Specific Semantic Search

The reporting system generates weekly industry reports by leveraging MongoDB's semantic search capabilities to gather relevant content for each report section. The scheduler runs automated jobs that create professional PDF reports for different industry/use case combinations.

**Key Components**:

1. **Report Templates**: 
   - MongoDB stores configurable templates by industry/use case
   - Each template defines report structure and section-specific prompts
   - Enables customized reports for Credit Rating, Investment Research, etc.

2. **Section-Specific Semantic Search**:
   - Each report section has its own targeted semantic query
   - Searches chunks collection using voyage-context-3 embeddings
   - Context accumulation ensures consistency across sections

3. **Scheduled Generation**:
   - Weekly automated reports (configurable by industry)
   - Tracks report metadata and file paths in scheduled_reports collection
   - Automatic cleanup of old reports (keeps last 7)

4. **PDF Output**:
   - Professional reports using ReportLab
   - Storage-agnostic output (local filesystem or cloud)
   - Includes key metrics, analysis sections, and disclaimer

**MongoDB Value for Reports**:
- **Unified Data Platform**: Report templates, document chunks, and vector embeddings all accessible from one database
- **Template Storage**: Flexible schema for industry-specific report structures
- **Semantic Search**: Fast content retrieval based on meaning for each report section
- **Metadata Tracking**: Report generation history and file management
- **Scalability**: Handle large document corpuses for comprehensive reports

## 📁 Project Structure

```
/
├── backend/                      # FastAPI Backend
│   ├── agents/                  # Agent Implementations
│   │   ├── agentic_rag_qa.py  # Agentic RAG Q&A system
│   │   ├── evaluator.py        # Document evaluation agent
│   │   ├── extractor.py        # Content extraction agent
│   │   ├── scanner.py         # Document discovery agent
│   │   ├── state.py          # State management
│   │   └── supervisor.py      # Orchestration agent
│   ├── api/                    # API Layer
│   │   ├── dependencies.py    # FastAPI dependencies
│   │   └── routes/           # API endpoints
│   │       ├── documents.py   # Document management
│   │       ├── ingestion.py   # Document ingestion
│   │       ├── qa.py         # Q&A endpoints
│   │       ├── reports.py     # Report generation
│   │       └── upload.py      # Document upload
│   ├── cloud/                 # Cloud Service Integrations
│   │   ├── aws/              # AWS Services
│   │   │   ├── bedrock/      # AWS Bedrock Integration
│   │   │   │   ├── claude_vision.py # Claude 3 vision
│   │   │   │   └── client.py        # AWS client setup
│   │   │   └── s3/           # S3 Storage Integration
│   │   │       ├── bucket_access.py # S3 operations
│   │   │       └── client.py        # S3 client setup
│   │   └── gdrive/           # Google Drive Integration
│   │       ├── gdrive_access.py  # Drive operations
│   │       └── simple_extraction.py # Public scraping
│   ├── config/               # Configuration Files
│   │   ├── industry_config.py # Industry mappings
│   │   └── storage_config.py  # Storage config
│   ├── data/                  # Demo Data
│   │   └── seed/             # Sample files
│   │       ├── docs/         # Sample documents
│   │       └── reports/      # Sample reports
│   ├── db/                    # Database Layer
│   │   ├── collections/      # MongoDB seed data
│   │   │   └── seeds/       # Config templates
│   │   ├── mongodb_connector.py # MongoDB connection
│   │   ├── vector_search.py    # Vector search
│   │   └── vector_search_index_creator.py # Index setup
│   ├── processors/            # Document Processing
│   │   └── document_processor.py # Main processor
│   ├── services/              # Background Services
│   │   ├── document_cache.py # Caching service
│   │   ├── report_generator.py # PDF generation
│   │   └── scheduler.py       # Report scheduler
│   ├── tools/                 # LangChain Tools
│   │   ├── document_tools.py  # Document handling
│   │   ├── embedding_tools.py # Embedding generation
│   │   └── vision_tools.py    # Vision processing
│   ├── vogayeai/              # VoyageAI Integration
│   │   └── context_embeddings.py # Context embeddings
│   ├── workflows/             # LangGraph Workflows
│   │   ├── ingestion_builder.py # Workflow builder
│   │   └── ingestion_workflow.py # Ingestion flow
│   ├── main.py               # FastAPI application
│   └── pyproject.toml        # Python dependencies
│
├── diagrams/                     # Architecture Diagrams
│   ├── 1_high_level_architecture.png
│   ├── 2_part1_ingestion_multiagent_supervisor.png
│   ├── 3_part1_multiagent_supervisor_pattern_explanation.png
│   ├── 4_part2_QandA_agentic_rag.png
│   ├── 5_part2_agentic_rag_pattern_explanation.png
│   └── 6_part3_scheduled_reports.png
│
├── environments/                 # Environment Configs
│   ├── prod.yaml               # Production config
│   └── staging.yaml            # Staging config
│
├── docker-compose.yml          # Docker services
├── Dockerfile.backend         # Backend container
├── makefile                  # Build commands
└── README.md                # This file
```

## 🔧 Configuration & Environment Variables

### Backend Configuration (.env)

⚠️ **CRITICAL**: The backend requires a `.env` file in the `backend/` directory. Without proper configuration, the demo will not work.

```bash
# Copy the example file (if available)
cp backend/.env.example backend/.env
# Then edit backend/.env with your actual values
```

#### Complete Environment Variables

```bash
# FSI Document Intelligence Environment Variables

# MongoDB Configuration (REQUIRED)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
DATABASE_NAME=document_intelligence
APP_NAME="ist.demo.document_intelligence.fsi"

# Collection Names
DOCUMENTS_COLLECTION=documents
CHUNKS_COLLECTION=chunks
ASSESSMENTS_COLLECTION=assessments
GRADINGS_COLLECTION=gradings
LOGS_COLLECTION=logs
AGENT_PERSONAS_COLLECTION=agent_personas

# Search Index
CHUNKS_VECTOR_INDEX="document_intelligence_chunks_vector_index"

# Document Storage Configuration
DOCUMENT_STORAGE_PATH=/docs
ALLOWED_INDUSTRIES=fsi,manufacturing,retail,healthcare,media,insurance
DEFAULT_INDUSTRY=fsi

# Upload Configuration
MAX_UPLOAD_SIZE_MB=1
ALLOWED_FILE_EXTENSIONS=pdf,docx,doc

# AWS Configuration (REQUIRED)
AWS_REGION=us-east-1
S3_BUCKET_NAME="industry-solutions-demos"
S3_BASE_PREFIX="industry/cross/document-intelligence"
# Optional: If using AWS SSO (recommended)
AWS_PROFILE=your-sso-profile-name

# Google Drive Configuration
GDRIVE_ROOT_FOLDER_ID=your-folder-id-here

# Bedrock Model Configuration (REQUIRED)
# Note: Use inference profile ID (us.*) for better availability
BEDROCK_MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_MAX_TOKENS=8192

# VoyageAI Configuration (REQUIRED)
VOYAGE_API_KEY=your-api-key-here
VOYAGE_MODEL=voyage-context-3

# Chunking Configuration
MAX_FILE_SIZE_MB=1
MAX_PAGES_PER_DOCUMENT=6
CHUNK_SIZE=2000
CHUNK_OVERLAP=0
```

#### Key Configuration Notes

- **MongoDB URI**: Replace with your Atlas connection string
- **AWS Profile**: Configure SSO with `aws configure sso` for secure access
- **VoyageAI Key**: Get from [VoyageAI](https://www.voyageai.com/)
- **Google Drive**: Use folder ID from shareable link (optional)
- **Bedrock Model**: Claude 3.5 Sonnet v2 with vision capabilities

## 📄 MongoDB Collections

The system uses 15 MongoDB collections organized by function:

> 📁 **Sample Data**: See `backend/db/collections/seeds/` for JSON seed files to populate these collections for local development.

### Core Document Processing (4 collections)
1. **chunks**: Document chunks with embeddings - stores text segments with voyage-context-3 vectors
2. **documents**: Document metadata - tracks processing status and source information  
3. **assessments**: Document evaluation results - stores relevance scores and processing decisions
4. **workflows**: Ingestion workflow tracking - monitors multi-agent processing state

### Q&A and Memory (5 collections)
5. **gradings**: Document relevance grading - binary scores from Q&A retrieval assessment
6. **logs_qa**: Q&A session logs - tracks agentic RAG workflow steps and decisions
7. **agent_personas**: Use-case specific AI configurations - stores prompts and capabilities per industry
8. **checkpoint_writes_aio**: LangGraph state writes - async persistence for conversation state
9. **checkpoints_aio**: Conversation memory - stores thread-based dialogue history

### Reports (2 collections)
10. **scheduled_reports**: Generated report metadata - tracks PDF locations and generation history
11. **report_templates**: Report structure templates - defines sections and prompts by use case

### Configuration (3 collections)
12. **buckets**: S3 bucket configurations - stores AWS bucket paths and access settings
13. **gdrive**: Google Drive configurations - public folder IDs for document scraping
14. **industry_mappings**: Industry classifications - maps industries to relevant topics and keywords

### System (1 collection)
15. **logs**: Workflow execution logs - INFO level logs for monitoring agent decisions

## 📦 Dependencies

### Backend (Python >=3.13)

From `backend/pyproject.toml`:

```python
# Core Framework
pymongo>=4.10.1              # MongoDB driver
python-dotenv>=1.0.1         # Environment variables
fastapi>=0.115.4             # API framework
uvicorn>=0.32.0              # ASGI server
pydantic>=2.0.0              # Data validation
python-multipart>=0.0.6      # File upload support

# AI/LLM Framework
langgraph>=0.2.0             # Agent orchestration
langchain>=0.3.0             # LLM framework
langchain-mongodb>=0.2.0     # MongoDB integration
langchain-aws>=0.2.0         # AWS Bedrock integration
langchain-voyageai>=0.1.0    # VoyageAI integration
langgraph-store-mongodb>=0.1.0     # State storage
langgraph-checkpoint-mongodb>=0.1.0 # Memory persistence

# Embeddings
voyageai>=0.3.2              # Context-aware embeddings

# Document Processing
pdf2image>=1.16.3            # PDF to image conversion
python-docx>=1.1.0           # DOCX file handling
pillow>=10.0.0               # Image processing
requests>=2.31.0             # HTTP client (Google Drive)

# AWS Services
boto3>=1.35.70               # AWS SDK
botocore>=1.35.70            # AWS core

# Report Generation
schedule>=1.2.0              # Task scheduling
matplotlib>=3.10.6           # Charts and graphs
reportlab>=4.4.4             # PDF generation
```

### System Dependencies

Required for document processing (install via system package manager):

```bash
# Ubuntu/Debian
apt-get install poppler-utils  # PDF processing
apt-get install libreoffice    # DOC/DOCX conversion

# macOS
brew install poppler           # PDF processing
brew install libreoffice       # DOC/DOCX conversion
```

## 📂 Document Sources

The system supports ingesting documents from multiple sources:

> **🔒 Security Note**: This repository includes only example configuration files (`.example.json`) in `backend/db/collections/seeds/`. You must create your own configuration files with real S3 bucket names and Google Drive folder IDs. Never commit these real configuration files to public repositories. See `backend/db/collections/seeds/README.md` for setup instructions.

> **🎯 Demo Data**: Sample documents and reports are available in `backend/data/seed/` for testing:
> - **Documents**: FSI use case examples (credit rating, KYC, loan origination, etc.) in `docs/`
> - **Reports**: Pre-generated PDF reports for each use case in `reports/`
> - **Initial State**: Configuration for demo scenarios in `documents_initial_state_dict.json`

### 1. Local Files (Docker Volume)
Documents can be uploaded via API and stored in the Docker volume:
```bash
# Upload documents
curl -X POST http://localhost:8080/api/upload/documents \
  -F "files=@document.pdf" \
  -F "industry=fsi" \
  -F "use_case=credit_rating"

# Available use cases: credit_rating, payment_processing_exception, investment_research, kyc_onboarding, loan_origination

# List uploaded documents in an industry/use_case
curl "http://localhost:8080/api/upload/documents/fsi?use_case=credit_rating"

# Delete specific document in an industry/use_case
curl -X DELETE "http://localhost:8080/api/upload/documents/fsi/document.pdf?use_case=credit_rating"

# Delete all documents in an industry/use_case folder
curl -X DELETE "http://localhost:8080/api/upload/documents/fsi?use_case=credit_rating"

# Ingest from local storage
curl -X POST http://localhost:8080/api/ingestion/start \
  -H "Content-Type: application/json" \
  -d '{
    "source_paths": ["@local@/docs/fsi/credit_rating"],
    "workflow_id": "local_fsi_ingestion"
  }'
```

### 2. AWS S3 Buckets
S3 bucket configuration is stored in MongoDB `buckets` collection. First, create your configuration from the example:
```bash
# Copy the example file
cp backend/db/collections/seeds/document_intelligence.buckets.example.json backend/db/collections/seeds/document_intelligence.buckets.json

# Edit the file to add your S3 bucket details
# Then import the configuration
mongoimport --uri "$MONGODB_URI" --collection buckets --file backend/db/collections/seeds/document_intelligence.buckets.json --jsonArray
```

### 3. Google Drive Folders
Google Drive folder configuration is stored in MongoDB `gdrive` collection. The system uses public folder web scraping (no API keys needed). First, create your configuration from the example:
```bash
# Copy the example file
cp backend/db/collections/seeds/document_intelligence.gdrive.example.json backend/db/collections/seeds/document_intelligence.gdrive.json

# Edit the file to add your Google Drive folder IDs
# Then import the configuration
mongoimport --uri "$MONGODB_URI" --collection gdrive --file backend/db/collections/seeds/document_intelligence.gdrive.json --jsonArray
```

#### Google Drive Structure:
```
📁 Document Intelligence Demo/
└── 📁 fsi/
    ├── 📁 credit_rating/
    ├── 📁 investment_research/
    ├── 📁 kyc_onboarding/
    ├── 📁 loan_origination/
    └── 📁 payment_processing_exception/
```

#### Google Drive Usage:
```bash
# Ingest from Google Drive FSI folder
curl -X POST http://localhost:8080/api/ingestion/start \
  -H "Content-Type: application/json" \
  -d '{
    "source_paths": ["@gdrive@fsi/credit_rating"],
    "workflow_id": "gdrive_fsi_ingestion"
  }'
```

### 4. Industry and Topic Mappings
The system uses context-aware document assessment based on industry and topic. This file is included in the repository and safe to use directly:
```bash
# Import industry mappings (no sensitive data - safe to use as-is)
mongoimport --uri "$MONGODB_URI" --collection industry_mappings --file backend/db/collections/seeds/document_intelligence.industry_mappings.json --jsonArray
```

#### S3 Document Sources by Industry:
Configure your own S3 bucket structure following this pattern:
- **FSI**: `s3://YOUR-BUCKET/your-path/fsi/`
- **Healthcare**: `s3://YOUR-BUCKET/your-path/healthcare/`
- **Insurance**: `s3://YOUR-BUCKET/your-path/insurance/`
- **Manufacturing**: `s3://YOUR-BUCKET/your-path/manufacturing/`
- **Media**: `s3://YOUR-BUCKET/your-path/media/`
- **Retail**: `s3://YOUR-BUCKET/your-path/retail/`

#### S3 Usage Examples
```bash
# Ingest from S3 FSI folder
curl -X POST http://localhost:8080/api/ingestion/start \
  -H "Content-Type: application/json" \
  -d '{
    "source_paths": ["@s3@fsi"],
    "workflow_id": "s3_fsi_ingestion"
  }'

# Ingest from specific S3 subfolder with use case
curl -X POST http://localhost:8080/api/ingestion/start \
  -H "Content-Type: application/json" \
  -d '{
    "source_paths": ["@s3@fsi/credit_rating"],
    "workflow_id": "s3_fsi_credit_rating"
  }'

# Mix local and S3 sources in one workflow
curl -X POST http://localhost:8080/api/ingestion/start \
  -H "Content-Type: application/json" \
  -d '{
    "source_paths": [
      "@local@/docs/fsi/credit_rating",
      "@s3@fsi/reports"
    ],
    "workflow_id": "mixed_sources_ingestion"
  }'
```

#### AWS Authentication for S3
The system uses AWS SSO for authentication. No access keys required:
1. Configure AWS SSO: `aws configure sso`
2. Login: `aws sso login --profile your-profile`
3. Set environment variable: `export AWS_PROFILE=your-profile`

### Source Path Format
All source types use a consistent prefix pattern for clarity:
- **Local files**: `@local@/docs/{industry}/{use_case}`
- **S3 files**: `@s3@{industry}` or `@s3@{industry}/{subfolder}`
- **Google Drive**: `@gdrive@{industry}/{use_case}`
- All three sources can be mixed in the same ingestion workflow

Document paths stored in MongoDB include full source information:
- Local: `@local@/path/to/file.pdf`
- S3: `@s3@bucket-name/path/to/file.pdf`
- Google Drive: `@gdrive@industry/use_case/file.pdf`

#### Mixed Source Example:
```bash
curl -X POST http://localhost:8080/api/ingestion/start \
  -H "Content-Type: application/json" \
  -d '{
    "source_paths": [
      "@local@/docs/fsi/credit_rating",
      "@s3@fsi/reports",
      "@gdrive@fsi/compliance"
    ],
    "workflow_id": "mixed_all_sources"
  }'
```

## 🎯 Context-Aware Document Assessment

The system evaluates documents based on their industry and topic context extracted from the source path:

### How It Works
1. **Path Analysis**: Extracts industry and topic from source paths
   - Example: `@s3@fsi/credit_rating` → Industry: "financial services", Topic: "credit rating"
   
2. **Relevance Scoring**: Documents are evaluated against:
   - Industry relevance (e.g., is this a financial services document?)
   - Topic relevance (e.g., is this about credit ratings?)
   - Documents matching EITHER criteria are accepted

3. **Strict Filtering**: Automatically rejects:
   - Food receipts, personal documents, entertainment content
   - Documents with no business relevance to the context
   - Test or sample documents

### Supported Industries
- **fsi**: Financial Services
- **healthcare**: Healthcare
- **insurance**: Insurance
- **manufacturing**: Manufacturing
- **media**: Media and Entertainment
- **retail**: Retail

## 🛠️ Development Commands

### UV 
```bash
# Ensure you are over /backend directory
cd backend

# Run locally
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### Docker
```bash
# Build and run all services
docker-compose up --build

# Run backend only
docker-compose up document-intelligence-backend
```

## 🌐 API Documentation

The system provides comprehensive API documentation through FastAPI's automatic documentation features:

### 📖 Interactive API Documentation
- **Swagger UI**: Navigate to `{URL}/docs` (e.g., `http://localhost:8080/docs`)
- **ReDoc**: Alternative documentation at `{URL}/redoc`

### 🔑 Key API Categories
- **Document Management**: Upload, list, and manage documents
- **Ingestion Workflows**: Start and monitor document processing
- **Q&A System**: Query documents with agentic RAG
- **Report Generation**: Generate and retrieve scheduled reports
- **System Health**: Status and configuration endpoints

> **💡 Tip**: The Swagger UI at `/docs` provides an interactive interface where you can:
> - Explore all available endpoints
> - View request/response schemas
> - Test API calls directly from your browser
> - See real-time responses and error codes

## 🧠 Memory and Session Management

The system implements sophisticated conversation persistence using MongoDB and LangGraph's checkpointing system:

### 📝 Conversation Memory Architecture
- **Thread-Based Sessions**: Each conversation has a unique `thread_id` for state isolation
- **Async Checkpointing**: Non-blocking state persistence using `checkpoint_writes_aio` collection
- **State Recovery**: Automatic restoration of conversation context across requests
- **Memory Types**:
  - **Working Memory**: Active conversation state in LangGraph
  - **Long-term Memory**: Persisted checkpoints in MongoDB
  - **Session Metadata**: Thread IDs, timestamps, and user context

### 🔄 How It Works
1. **Session Initiation**: Generate unique session ID when user starts Q&A
2. **State Checkpointing**: After each agent decision, state is saved to MongoDB
3. **Context Retrieval**: Previous messages and decisions loaded for continuity
4. **Memory Cleanup**: Optional session cleanup via API endpoints

### 💾 MongoDB Collections Used
- **`checkpoints_aio`**: Stores complete conversation states
- **`checkpoint_writes_aio`**: Handles async write operations
- **`logs_qa`**: Tracks session events and agent decisions
- **`gradings`**: Preserves document relevance assessments per session

### 🚀 Usage Example
```python
# Start new session
session_id = "user-123-session-456"

# Query with memory
response = await qa_system.answer_with_agentic_rag(
    query="What is the credit rating?",
    thread_id=session_id  # Enables conversation memory
)

# Follow-up uses same session
follow_up = await qa_system.answer_with_agentic_rag(
    query="Why did it change?",  # Understands context from previous question
    thread_id=session_id
)
```

## 🎯 Key Architectural Decisions

### Why Three Different Patterns?

1. **Ingestion**: Documents need systematic, quality-controlled processing
   - Deterministic workflow ensures data integrity
   - Quality gates prevent irrelevant content
   - Sequential processing ensures completeness

2. **Q&A**: User queries need adaptive, intelligent responses
   - Dynamic routing based on query complexity
   - Self-correction through document grading
   - Iterative improvement through query rewriting

3. **Reports**: Automated content generation needs targeted context retrieval
   - Section-specific semantic search for accurate context
   - Context accumulation for report consistency
   - Scheduled generation with fallback mechanisms

### MongoDB Value Proposition

1. **Unified Data Platform**: Store structured metadata, unstructured documents, and vector embeddings in a single database
2. **Vector Search**: Lightning-fast semantic search across document collections
3. **Multi-Collection Intelligence**: Specialized collections for workflows, assessments, personas, and more
4. **Real-time Updates**: Live workflow tracking and conversation memory persistence
5. **Scalability**: Atlas handles enterprise-scale document processing with automatic scaling

## 🔒 Security Notes

- AWS credentials are mounted as read-only in Docker
- API keys should be stored in environment variables
- MongoDB connection strings should be secured

## 🔧 Troubleshooting

### Common Issues

1. **"MongoDB URI must be provided" Error**
   - Ensure `backend/.env` file exists with `MONGODB_URI` set
   - Check that Docker Compose is reading the env file correctly

2. **"VOYAGE_API_KEY is required" Error**
   - Add your VoyageAI API key to `backend/.env`
   - Sign up at https://www.voyageai.com if you don't have one

3. **AWS Authentication Errors**
   - Configure AWS SSO: `aws configure sso`
   - Login: `aws sso login --profile your-profile`
   - Set `AWS_PROFILE` in `backend/.env`

4. **Document Processing Fails**
   - Check `MAX_FILE_SIZE_MB` and `MAX_PAGES_PER_DOCUMENT` in `.env`
   - Ensure Docker has enough memory allocated (at least 4GB)

5. **S3/Google Drive Not Working**
   - Import the MongoDB seed configurations (see Database Setup)
   - Check that bucket/folder IDs are correct in your JSON files

6. **"Invocation of model ID with on-demand throughput isn't supported" Error**
   - Use the inference profile ID instead of direct model ID
   - For Claude 3.5 Sonnet v2: Use `us.anthropic.claude-3-5-sonnet-20241022-v2:0`
   - Check available inference profiles: `aws bedrock list-inference-profiles`

## 📚 References

- [MongoDB Atlas Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [VoyageAI: voyage-context-3](https://blog.voyageai.com/2025/07/23/voyage-context-3/)
- [Concept: Multi-agent systems](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [LangGraph Agentic RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
- [LangChain Supervisor Pattern](https://docs.langchain.com/oss/python/langchain/supervisor)
- [AWS Bedrock Claude 3.5 Sonnet](https://aws.amazon.com/bedrock/claude/)

## 📄 License

See [LICENSE](LICENSE) file for details.
