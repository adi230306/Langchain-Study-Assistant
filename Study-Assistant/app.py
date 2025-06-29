from flask import Flask, request, jsonify, abort
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename  # Add this import
import os
from dotenv import load_dotenv
import tempfile
import shutil
from flask_cors import CORS
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
FAISS_INDEX_DIR = "data/faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt"]

# Initialize components
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize FAISS vector store
vectorstore = None

def initialize_vectorstore():
    """Initialize or load the FAISS vectorstore"""
    global vectorstore
    try:
        if os.path.exists(FAISS_INDEX_DIR) and os.listdir(FAISS_INDEX_DIR):
            vectorstore = FAISS.load_local(
                FAISS_INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("Loaded existing FAISS index")
        else:
            os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
            print("Created new FAISS index directory")
            vectorstore = None
    except Exception as e:
        print(f"Error initializing vectorstore: {e}")
        vectorstore = None

initialize_vectorstore()

def process_uploaded_files(files):
    """Process uploaded study material files"""
    documents = []
    temp_dir = tempfile.mkdtemp()
    
    for file in files:
        # Initialize loader as None
        loader = None
        if not any(file.filename.lower().endswith(ext) for ext in SUPPORTED_FILE_TYPES):
            continue
            
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        try:
            if file.filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.filename.lower().endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file.filename.lower().endswith(".txt"):
                loader = TextLoader(file_path)
            
            # Only load if valid loader was created
            if loader:
                documents.extend(loader.load())
                
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    shutil.rmtree(temp_dir)
    return documents

@app.errorhandler(HTTPException)
def handle_exception(e):
    return jsonify({
        "error": e.name,
        "message": e.description
    }), e.code

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and document processing"""
    # Check if files were submitted
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
        
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No selected files"}), 400
    
    # Initialize document processing
    processed_docs = []  # Stores loaded documents
    temp_dir = tempfile.mkdtemp()
    
    for file in files:
        try:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(temp_dir, filename)
                file.save(filepath)
                
                # Load document based on file type
                if filename.lower().endswith('.pdf'):
                    loader = PyPDFLoader(filepath)
                elif filename.lower().endswith('.docx'):
                    loader = Docx2txtLoader(filepath)
                elif filename.lower().endswith('.txt'):
                    loader = TextLoader(filepath)
                
                processed_docs.extend(loader.load())
                
        except Exception as e:
            app.logger.error(f"Error processing {file.filename}: {str(e)}")
            continue
    
    # Clean up temp files
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    if not processed_docs:
        return jsonify({"error": "No valid content extracted"}), 400
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(processed_docs)
    
    # Update vectorstore
    global vectorstore
    if vectorstore is None:
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )
    else:
        vectorstore.add_documents(splits)
    
    vectorstore.save_local(FAISS_INDEX_DIR)
        
    return jsonify({
        "message": "Files processed successfully",
        "total_chunks": len(splits),
        "files_processed": len(processed_docs),
        "average_chunk_size": len(splits)//max(1, len(processed_docs))
    })

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx', 'txt'}

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        question = data.get('question')
        
        if not question or not isinstance(question, str):
            return jsonify({"error": "Invalid question format"}), 400
        
        # Validate vectorstore exists
        if vectorstore is None:
            return jsonify({
                "answer": "No documents have been uploaded yet. Please upload study materials first.",
                "status": "error"
            }), 200
            
        # Process question
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        prompt_template = """You are a helpful study assistant. Follow these rules:
                1. FIRST check if the question relates to the provided context
                2. If relevant, answer using ONLY the context with citations
                3. If unrelated, provide a general knowledgeable answer
                4. Never say you can't answer - always respond helpfully
                5. Answer the question in detail.

                Context: {context}
                Question: {question}

                Thought process: Let me think step by step:"""
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | PromptTemplate.from_template(prompt_template)
            | model
            | StrOutputParser()
        )
        
        answer = chain.invoke(question)
        
        return jsonify({
            "answer": answer,
            "status": "success"
        })
        
    except Exception as e:
        app.logger.error(f"Error processing question: {str(e)}")
        return jsonify({
            "error": "Failed to process question",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "vectorstore_ready": vectorstore is not None,
        "documents_indexed": len(vectorstore.index_to_docstore_id) if vectorstore else 0
    })

if __name__ == '__main__':
    app.run(debug=True)