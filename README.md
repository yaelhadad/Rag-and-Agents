# Simplified RAG System with Vector Storage

A streamlined document analysis system using LangChain with FAISS vector storage, automatic metadata extraction, and intelligent system comparison capabilities.

## 🚀 Features

- **Vector Storage**: FAISS-based vector storage with automatic persistence
- **Metadata Export**: Automatic export of chunk metadata to JSON for analysis
- **Smart Caching**: Rebuilds vectorstore only when needed
- **System Comparison**: Compare features and capabilities across multiple systems
- **Memory Support**: Conversation memory for contextual responses
- **Environment-Based Configuration**: Secure API key management
- **Automatic Chunking**: Intelligent document splitting with metadata preservation

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see `requirements.txt`)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd multi_agent_rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4-turbo-preview
   TEMPERATURE=0.7
   MAX_TOKENS=4000
   DEBUG=False
   ```

## 📖 Usage

### Basic Usage

```python
from core.agents import main

# Run the system
main()
```

### What the System Does

1. **Loads Documents**: Automatically loads all `.txt` files from the `data/` directory
2. **Creates Chunks**: Splits documents into logical chunks with metadata
3. **Builds Vectorstore**: Creates FAISS vector storage (saved to `faiss_index/`)
4. **Exports Metadata**: Saves chunk metadata to `vector_metadata.json`
5. **Answers Questions**: Uses the vectorstore to answer questions about the systems

### Example Questions

The system can answer questions like:
- "What can customers do in the restaurant reservation system?"
- "Compare the use of databases in both systems?"
- "What are the main differences between admin and staff roles?"

### Generated Files

After running the system, you'll get:
- `faiss_index/` - FAISS vector storage directory
- `vector_metadata.json` - Metadata for all document chunks

### Tools Available

- **Summarize**: Summarizes long text using LLM
- **Star**: Adds asterisks around answers for emphasis

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | Model to use | `gpt-4-turbo-preview` |
| `TEMPERATURE` | Creativity level | `0.7` |
| `MAX_TOKENS` | Max tokens per response | `4000` |

### Data Directory

Place your `.txt` files in the `data/` directory. The system will automatically:
- Load all `.txt` files
- Split them into chunks
- Extract metadata (source file, title, etc.)
- Build vector embeddings

## 🧪 Testing

```bash
# Run the main system
python core/agents.py

# Check generated files
ls faiss_index/
cat vector_metadata.json
```

## 📊 Metadata Analysis

The system exports detailed metadata for each chunk:
- `source`: Source file name
- `title`: Chunk title/heading
- Additional metadata as extracted

You can analyze the `vector_metadata.json` file to understand:
- How documents were chunked
- What metadata was extracted
- Document structure and organization

## 🔄 Rebuilding

To force rebuild of the vectorstore and metadata:
1. Delete `faiss_index/` directory
2. Delete `vector_metadata.json` file
3. Run the system again

The system will automatically detect missing files and rebuild everything.

## 🚀 Optimization Features

- **Smart Caching**: Only rebuilds when files are missing
- **Efficient Embeddings**: Uses OpenAI's text-embedding-ada-002
- **Memory Management**: Conversation memory for context
- **Error Handling**: Graceful handling of missing files and API errors

## 📝 Example Output

```
Loaded 8 chunks from 2 files: ['system_design_restaurant_ordering.txt', 'system_design_daily_dev_team_task_tracker.txt']
Vectorstore and metadata saved.
Metadata fields: {'source', 'title'}

Q1: Compare the use of databases in both systems?

[File: system_design_restaurant_ordering.txt, Title: Core Components]
FastAPI backend, React UI, PostgreSQL database, Redis for real-time updates.

[File: system_design_daily_dev_team_task_tracker.txt, Title: Core Components]  
FastAPI backend, React frontend, Mysql database, and optional Redis cache.

A1: Both systems use different databases: PostgreSQL vs MySQL...
