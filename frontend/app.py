import streamlit as st
import requests
import json
import uuid
import pandas as pd
from typing import List, Dict, Any

# --- Config ---
# IMPORTANT: In Docker, the frontend container must call the backend via the
# Docker Compose service name 'backend', not 'localhost'.
# For local development, override this with an environment variable.
import os
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000/api/v1")

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- Page Config ---
st.set_page_config(
    page_title="Antigravity RAG - Chat with PDF",
    page_icon="👾",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.title("📂 Document Management")
    st.write("Upload PDFs to index them into the FAISS vector store.")
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("🚀 Index Document"):
            with st.spinner("Processing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{BACKEND_URL}/ingestion/upload", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Indexed: {data['filename']}")
                        # Refresh file list
                        st.session_state.uploaded_files.append({"filename": data['filename'], "doc_id": data['doc_id']})
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {str(e)}")

    st.divider()
    
    # List documents
    if st.button("🔄 Refresh Document List"):
        try:
            response = requests.get(f"{BACKEND_URL}/ingestion/documents")
            if response.status_code == 200:
                 st.session_state.uploaded_files = response.json()
            else:
                 st.error("Failed to load document list.")
        except Exception as e:
            st.error(f"Failed to connect to backend: {str(e)}")

    if st.session_state.uploaded_files:
        st.write("### Managed Documents")
        for doc in st.session_state.uploaded_files:
            st.info(f"📄 {doc['filename']}")
    else:
        st.write("No documents indexed yet.")

# --- Main App ---
st.title("👾 Antigravity RAG System")
st.caption("A production-grade, zero-hallucination PDF chat interface powered by Gemini 3.1 Pro and Hybrid FAISS/BM25 retrieval.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message:
            with st.expander("📚 Citations"):
                for cit in message["citations"]:
                    st.write(f"**[SOURCE_{cit['source_n']}]** {cit['doc_name']} (Page {cit['page_num']})")

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        citations = []
        
        # We can implement streaming here for better UX
        try:
             # Use the /query endpoint for synchronous response in Streamlit for now
             # or /stream for SSE.
             payload = {
                 "question": prompt,
                 "session_id": st.session_state.session_id,
                 "use_hyde": True
             }
             
             # Call backend
             response = requests.post(f"{BACKEND_URL}/chat/query", json=payload)
             
             if response.status_code == 200:
                 data = response.json()
                 full_response = data["answer"]
                 message_placeholder.markdown(full_response)
                 
                 # Show Citations
                 if data["citations"]:
                     citations = data["citations"]
                     with st.expander("📚 Citations"):
                         for cit in citations:
                             st.write(f"**[SOURCE_{cit['source_n']}]** {cit['doc_name']} (Page {cit['page_num']})")
                 
                 # Save to history
                 st.session_state.messages.append({
                     "role": "assistant", 
                     "content": full_response, 
                     "citations": citations
                 })
             else:
                 st.error(f"Backend Error: {response.text}")
                 
        except Exception as e:
             st.error(f"Failed to connect to backend: {str(e)}")

# Footer
st.divider()
st.caption("System Status: Operational | Powered by Google Gemini & FAISS")
