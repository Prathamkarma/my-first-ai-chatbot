import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
def process_document(file_path):
    """Reads a PDF and splits it into smaller chunks."""
    
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Setup the Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # 3. Chop the document into chunks
    chunks = text_splitter.split_documents(documents)
    
    return chunks