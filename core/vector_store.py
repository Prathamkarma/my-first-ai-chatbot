import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# We define a physical folder name to save our database
DB_PATH = "vector_db"

def get_embeddings():
    """Helper function to load our mathematical model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(chunks):
    """Converts text chunks into math vectors and saves them to the HARD DRIVE."""
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # NEW: Save the database to a physical folder
    vector_store.save_local(DB_PATH)
    
    return vector_store

def load_existing_vector_store():
    """Checks if a database folder already exists on the hard drive and loads it."""
    if os.path.exists(DB_PATH):
        embeddings = get_embeddings()
        # Load it back into memory! 
        # (allow_dangerous_deserialization=True is required by LangChain when loading local files you trust)
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None