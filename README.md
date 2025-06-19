# Multi-Agent RAG System

A sophisticated document analysis and question-answering system using multiple AI agents with advanced optimization features.

## 🚀 Features

- **Enhanced Document Processing**: Automatic metadata extraction and intelligent document chunking
- **Multi-Agent Architecture**: Specialized agents for different document types
- **Cost Optimization**: Token usage tracking and cost management
- **Smart Caching**: Automatic index rebuilding only when documents change
- **Environment-Based Configuration**: Secure API key management
- **Advanced Search**: Multiple search strategies with relevance scoring

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
   ```bash
   python setup_env.py
   ```
   
   Or manually create a `.env` file:
   ```env
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4-turbo-preview
   TEMPERATURE=0.7
   MAX_TOKENS=4000
   DEBUG=False
   ```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test configuration and imports
python simple_test.py

# Test the full agent system
python test_agent.py

# Test configuration only
python test_config.py
```

## 📖 Usage

### Basic Usage

```python
from core.agents import EnhancedDocumentAgent

# Create an agent for your document
agent = EnhancedDocumentAgent("LegalAssistant", "path/to/your/document.txt")

# Ask questions
answer = agent.ask("What are the main clauses in this document?")
print(answer)
```

### Advanced Usage

```python
from core.agents import EnhancedDocumentAgent
from core.token_tracker import CostOptimizer

# Create agent with custom index path
agent = EnhancedDocumentAgent(
    agent_name="ContractAnalyzer",
    filepath="contracts/lease_agreement.txt",
    index_path="custom_index_path"
)

# Get document structure
structure = agent.get_metadata_info()
print(f"Document has {structure['document_structure']['clauses']} clauses")

# Ask specific questions
questions = [
    "What is the security deposit amount?",
    "What are the termination conditions?",
    "Find clause 5"
]

for question in questions:
    answer = agent.ask(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### Cost Optimization

```python
from core.token_tracker import CostOptimizer

# Initialize cost optimizer
optimizer = CostOptimizer(
    model="gpt-4",
    max_context_tokens=2000,
    save_history=True
)

# Track usage
optimizer.track_usage(
    session_id="session_1",
    input_tokens=1500,
    output_tokens=500,
    session_name="contract_analysis"
)

# Get usage report
optimizer.print_usage_report()
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | - | Yes |
| `OPENAI_MODEL` | Model to use | `gpt-4-turbo-preview` | No |
| `TEMPERATURE` | Creativity level (0.0-1.0) | `0.7` | No |
| `MAX_TOKENS` | Maximum tokens per response | `4000` | No |
| `DEBUG` | Enable debug mode | `False` | No |

### Optimization Settings

- **Model Selection**:
  - `gpt-3.5-turbo`: Faster, cheaper responses
  - `gpt-4`: More accurate, detailed responses
  
- **Temperature Tuning**:
  - `0.0-0.3`: Factual, consistent responses
  - `0.4-0.7`: Balanced creativity and accuracy
  - `0.8-1.0`: More creative, varied responses

## 📊 Performance Optimization

### 1. **Token Management**
- Monitor token usage with `CostOptimizer`
- Use context optimization for long documents
- Implement token budgeting for cost control

### 2. **Caching Strategy**
- Indexes are automatically cached and rebuilt only when needed
- File checksums prevent unnecessary rebuilds
- Metadata is preserved for quick access

### 3. **Search Optimization**
- Multiple search strategies available:
  - General document search
  - Clause-specific search
  - Keyword-based search with metadata
  - Similarity search

### 4. **Memory Management**
- Conversation memory for context continuity
- Automatic memory cleanup
- Configurable memory limits

## 🔒 Security

- API keys are stored in environment variables
- `.env` files are excluded from version control
- No sensitive data is logged
- Secure token handling

## 📁 Project Structure

```
multi_agent_rag/
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py        # Environment variable handling
├── core/                  # Core functionality
│   ├── agents.py          # Main agent implementation
│   └── token_tracker.py   # Cost optimization
├── data/                  # Document storage
│   ├── pdfs/             # PDF documents
│   └── qna/              # Q&A datasets
├── index_*/               # Generated indexes
├── tests/                 # Test files
├── utils/                 # Utility functions
├── setup_env.py           # Environment setup script
├── test_agent.py          # Agent testing
├── simple_test.py         # Import testing
└── CONFIGURATION.md       # Configuration guide
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Make sure you're in the correct directory
   cd multi_agent_rag
   python simple_test.py
   ```

2. **API Key Issues**:
   ```bash
   # Check your environment variables
   python setup_env.py
   ```

3. **File Not Found**:
   ```bash
   # Ensure data files exist
   ls data/
   ```

4. **Memory Issues**:
   - Reduce `max_context_tokens`
   - Use smaller document chunks
   - Enable debug mode for detailed logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the troubleshooting section
2. Review the configuration guide
3. Run the test scripts
4. Create an issue with detailed error information
