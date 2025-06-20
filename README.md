# Document Summarization with Retrieval-Augmented Generation (RAG)

## Overview

This project implements a comprehensive document summarization system using Retrieval-Augmented Generation (RAG) technology. The system intelligently processes documents in multiple formats (PDF, TXT, Markdown), extracts relevant information through semantic retrieval, and generates high-quality summaries using state-of-the-art language models.

## Key Features

- **Multi-format Support**: Process PDF, TXT, and Markdown documents seamlessly
- **Semantic Chunking**: Intelligent text splitting with overlapping windows for context preservation
- **Advanced Retrieval**: Efficient embedding and similarity search using SentenceTransformers and FAISS
- **BART-powered Summarization**: Generate concise, coherent summaries with Facebook's BART model
- **Comprehensive Evaluation**: Automated assessment of summary quality including fluency, coverage, and accuracy
- **Performance Metrics**: Track token usage, processing latency, and similarity scores
- **Batch Processing**: Handle multiple documents efficiently

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Sufficient RAM for model loading (4GB+ recommended)
- GPU support optional but recommended for faster processing

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd document-summarization-rag
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK Resources
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

## Project Structure

### Current Structure
```
project_root/
├── document_ingestion.py    # Document loading and chunking
├── retrieval.py            # Embedding and FAISS retrieval
├── summarization.py        # BART-based summary generation
├── main.py                 # Main pipeline orchestration
├── sample.pdf              # Sample document 1 (Generative AI in drug discovery)
├── sample2.pdf             # Sample document 2 (ArXiv paper)
├── sample3.pdf             # Sample document 3 (CNN/DailyMail article)
├── output_doc1.txt         # Generated summary for document 1
├── output_doc2.txt         # Generated summary for document 2
├── output_doc3.txt         # Generated summary for document 3
├── evaluation.txt          # Evaluation metrics for all documents
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .venv/                 # Virtual environment (excluded from git)
└── __pycache__/           # Python cache files (excluded from git)
```


## Usage

### Basic Usage
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Run the complete pipeline
python main.py
```

### Output Files
After running the pipeline, you'll find:

- **output_docX.txt**: Individual summary files containing:
  - Retrieved relevant chunks
  - Generated summary
  - Performance metrics (token counts, processing latency)
  - Similarity scores

- **evaluation.txt**: Comprehensive evaluation results including:
  - Fluency scores
  - Coverage metrics
  - Accuracy assessments
  - Overall quality ratings

### Customization Options

#### Modify Input Documents
Update the document paths in `main.py`:
```python
documents = [
    "path/to/your/document1.pdf",
    "path/to/your/document2.txt",
    "path/to/your/document3.md"
]
```

#### Adjust Processing Parameters
- **Chunk Size**: Modify in `document_ingestion.py` to change text splitting granularity
- **Retrieval Count**: Adjust number of retrieved chunks in `retrieval.py`
- **Summary Length**: Configure BART parameters in `summarization.py`

## Key Dependencies

| Library | Purpose |
|---------|---------|
| `langchain` | Document processing and chunking |
| `sentence-transformers` | Text embedding generation |
| `faiss-cpu` | Efficient similarity search |
| `transformers` | BART model for summarization |
| `torch` | Deep learning framework |
| `nltk` | Natural language processing utilities |
| `PyPDF2` | PDF document parsing |

See `requirements.txt` for complete dependency list with versions.

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB free disk space

### Recommended Requirements
- Python 3.9+
- 8GB+ RAM
- GPU with CUDA support (for faster processing)
- 5GB+ free disk space

## Performance Optimization

### For CPU-only Systems
- Reduce batch size in summarization
- Use smaller embedding models
- Process documents sequentially

### For GPU-enabled Systems
- Enable CUDA in torch installation
- Increase batch sizes
- Use larger, more accurate models

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce chunk size
   - Process fewer documents simultaneously
   - Use CPU-only mode

2. **Slow Processing**
   - Enable GPU acceleration
   - Reduce document size
   - Optimize chunk overlap

3. **Import Errors**
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Getting Help
- Check error logs in console output
- Verify all dependencies are installed correctly
- Ensure input documents are in supported formats

## Development

### Adding New Document Formats
1. Extend `document_ingestion.py` with new loader
2. Add format detection logic
3. Update supported formats in documentation

### Improving Summarization Quality
1. Experiment with different embedding models
2. Adjust chunk size and overlap parameters
3. Fine-tune BART model parameters
4. Implement custom evaluation metrics

## Version Control Setup

Create a `.gitignore` file:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual Environment
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
```

## Future Enhancements

- [ ] Support for more document formats (DOCX, HTML)
- [ ] Web interface for document upload
- [ ] Real-time processing capabilities
- [ ] Custom model fine-tuning
- [ ] Multi-language support
- [ ] API endpoint development
- [ ] Docker containerization

