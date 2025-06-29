import streamlit as st
import requests
import os
from io import BytesIO
from PIL import Image
import tempfile
import time

# Configuration
BACKEND_URL = "http://127.0.0.1:5000"  # Update if your backend runs elsewhere

# Page setup
st.set_page_config(
    page_title="Study Assistant", 
    page_icon="📚", 
    layout="wide"
)
st.title("📚 Personalized Study Assistant")
st.caption("Upload your study materials and ask questions")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar for uploads and system status
with st.sidebar:
    st.header("Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose study materials (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Upload button with enhanced status handling
    if st.button("📤 Upload Files", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing files..."):
                try:
                    files = []
                    for file in uploaded_files:
                        # Reset file pointer and prepare for upload
                        file.seek(0)
                        files.append(("files", (file.name, file.getvalue(), file.type)))
                    
                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        st.session_state.uploaded_files.extend(uploaded_files)
                        st.success(f"✅ Processed {response.json()['total_chunks']} document chunks")
                        time.sleep(1)  # Allow success message to display
                        st.rerun()
                    else:
                        error_msg = response.json().get("error", "Unknown error")
                        st.error(f"❌ Upload failed: {error_msg}")
                except requests.exceptions.RequestException as e:
                    st.error(f"🚨 Connection error: {str(e)}")
                except Exception as e:
                    st.error(f"⚠️ Unexpected error: {str(e)}")
        else:
            st.warning("Please select files first")

    # Display uploaded files
    st.divider()
    st.subheader("Uploaded Files")
    if st.session_state.uploaded_files:
        for file in st.session_state.uploaded_files:
            st.caption(f"• {file.name}")
    else:
        st.caption("No files uploaded yet")

    # System status
    st.divider()
    st.subheader("System Status")
    
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if response.status_code == 200:
            health = response.json()
            st.metric("📊 Documents Indexed", health.get("documents_indexed", 0))
            st.metric("🔌 Backend Status", "Connected ✅")
        else:
            st.error(f"Backend error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection failed: {str(e)}")

# Main chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # ... [existing user message handling code] ...
    
    # Get assistant response
    with st.spinner("🔍 Searching documents..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/ask",
                json={"question": prompt},
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer returned")
                
                # Add this block to handle "don't know" responses
                if "don't know" in answer.lower() or "no information" in answer.lower():
                    answer = "Here's what I know about this topic:\n\n" + answer
                    
            else:
                answer = f"Error: {response.json().get('message', 'Failed to get response')}"
                
        except requests.exceptions.RequestException as e:
            answer = f"Connection error: {str(e)}"
        except Exception as e:
            answer = f"Processing error: {str(e)}"
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Help section
with st.expander("ℹ️ How to use this assistant"):
    st.markdown("""
    1. **Upload documents** using the sidebar (PDF, Word, or Text)
    2. **Ask questions** about the content in the chat below
    3. **Review answers** with sources from your documents

    **Tips:**
    - Start with 1-2 documents to test
    - Ask specific questions for best results
    - Supported formats: PDF, DOCX, TXT
    """)

# Run with: streamlit run frontend.py