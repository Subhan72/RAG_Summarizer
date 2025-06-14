from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re

def load_document(file_path):
    """
    Load a document from the given file path and return its text content.
    Supports PDF, TXT, and Markdown formats. Filters out references, URLs, and metadata.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext in ['.txt', '.md']:
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF, TXT, or Markdown.")
    
    documents = loader.load()
    text = ' '.join([doc.page_content for doc in documents])
    text = re.sub(r'^\[\d+\].*?$|https?://[^\s]+|Corresponding author.*?$|Copyright.*?$', '', text, flags=re.MULTILINE)
    text = ' '.join(text.split())
    return text

def chunk_text(text, chunk_size=2000, chunk_overlap=400):
    """
    Split the text into overlapping chunks using a recursive character splitter.
    Default chunk size is 2000 characters with a 400-character overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks