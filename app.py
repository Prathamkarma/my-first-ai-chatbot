import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import shutil

# Import our custom backend modules
from core.document_processor import process_document
from core.vector_store import create_vector_store, load_existing_vector_store, DB_PATH
from core.rag_engine import answer_with_rag

# Load environment variables (API Key)
load_dotenv()

st.title("My RAG AI Chatbot 🤖 (V2.0)")

# 1. Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

# On startup, attempt to load an existing database from the hard drive
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_existing_vector_store()

# 2. The Sidebar: File Upload & Memory Management
with st.sidebar:
    st.header("Document Upload")
    
    # Check if we successfully loaded a database
    if st.session_state.vector_store is not None:
        st.success("Database Loaded from Hard Drive! 🧠")
        
        # Add a button to wipe the physical memory
        if st.button("Clear Database"):
            st.session_state.vector_store = None
            st.session_state.messages = [] # Wipe chat history
            if os.path.exists(DB_PATH):
                shutil.rmtree(DB_PATH) # Delete the physical folder
            st.rerun() 
    else:
        st.warning("No database found. Chatting in general mode.")

    # Allows both PDF and TXT file uploads
    uploaded_file = st.file_uploader("Upload a document to chat with it", type=["pdf", "txt"])
    
    # Process new file only if we don't have one loaded
    if uploaded_file and st.session_state.vector_store is None:
        with st.spinner("Processing, chunking, and saving to hard drive..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Run backend pipeline
            chunks = process_document(tmp_file_path)
            
            # Defensive Shield
            if len(chunks) == 0:
                st.error("⚠️ Could not extract any readable text from this file.")
                os.remove(tmp_file_path)
            else:
                st.session_state.vector_store = create_vector_store(chunks) 
                os.remove(tmp_file_path)
                st.success("Document processed and saved permanently!")
                st.rerun()

# 3. Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle New User Input (Fixed & Connected)
if prompt := st.chat_input("Ask a question..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Safely passes the vector_store even if it is None!
            stream_generator = answer_with_rag(prompt, st.session_state.vector_store, st.session_state.messages)
            response_text = st.write_stream(stream_generator)
                
    st.session_state.messages.append({"role": "assistant", "content": response_text})