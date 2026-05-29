import os
from groq import Groq

def answer_with_rag(query, vector_store, chat_history):
    """
    Intelligently routes the query: Uses document context if available and relevant,
    otherwise falls back to general conversation mode.
    """
    # 1. Format Chat History (Keep last 4 messages)
    history_text = ""
    for msg in chat_history[-4:]: 
        if msg["role"] == "user":
            history_text += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history_text += f"AI: {msg['content']}\n"
            
    # 2. Dynamic Prompt Generation based on Vector Store Availability
    if vector_store is not None:
        # Search documents for context
        docs = vector_store.similarity_search(query, k=3)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        rag_prompt = f"""You are a helpful, conversational AI assistant. 
        You have access to retrieved document context and previous conversation history.
        
        CRITICAL ROUTING RULES:
        1. If the user's question is a casual greeting (e.g., 'hi', 'hello', 'hey'), a personal intro, or a general knowledge question entirely unrelated to the uploaded documents, ignore the context and respond naturally using your own general knowledge.
        2. If the user's question asks about details within the document, use the retrieved context below to provide an accurate answer. If the context doesn't contain the information, say you don't know based on the document.

        Previous Conversation History:
        {history_text}

        Retrieved Document Context:
        {context_text}

        Current User Question:
        {query}
        """
    else:
        # FALLBACK: No document uploaded -> Act as a pure general-purpose chatbot
        rag_prompt = f"""You are a helpful, friendly, and smart AI assistant. 
        Chat naturally with the user and answer their general knowledge questions accurately.
        
        Previous Conversation History:
        {history_text}

        Current User Question:
        {query}
        """
    
    # 3. Call Groq with the dynamically routed prompt
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    completion_stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": rag_prompt}],
        stream=True
    )
    
    for chunk in completion_stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta