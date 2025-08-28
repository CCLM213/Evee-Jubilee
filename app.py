# app.py

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Evee - Your Financial Assistant",
    page_icon="📚",
    layout="centered"
)

# --- Page Title ---
st.title("📚 Evee's Magic Tome")

# --- Introduction Text ---
st.write("""
Welcome to the Magic Tome. Upload a company policy PDF and start asking questions!
""")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Upload your company's policy PDF",
    type="pdf",
    help="Please upload one PDF document containing your company's policies."
)

# This function will handle the RAG logic
def process_document(file_path):
    # 1. Load and Split Document 
    # We use a temporary file path to load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split the document into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 2. Create Vector Embeddings and Store in Chroma DB 
    # Use OpenAI's embedding model to create vectors
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    # Store the vectorized chunks in a Chroma vector store
    # This creates a searchable "database" of our document's content
    vector_store = Chroma.from_documents(texts, embeddings)

    # 3. Create a Retriever and QA Chain 
    # The retriever's job is to find the most relevant document chunks
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    
    # The LLM will generate the answer based on the retrieved context
    llm = ChatOpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    # We combine the retriever and LLM into a "chain"
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    return qa_chain

# --- Main Application Logic ---
if uploaded_file is not None:
    # Use a temporary file to handle the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner('Analyzing the document... This may take a moment.'):
        # Create and store the QA chain in the session state
        st.session_state.qa_chain = process_document(tmp_file_path)
        st.success("Document analyzed successfully. You can now ask questions.")
    
    # Clean up the temporary file
    os.remove(tmp_file_path)

# --- Chat Input and Response ---
if 'qa_chain' in st.session_state:
    user_question = st.chat_input("Ask a question about your document...")
    if user_question:
        st.write(f"**You asked:** {user_question}")
        with st.spinner("Searching the tome for an answer..."):
            # Get the answer from the QA chain [cite: 23]
            answer = st.session_state.qa_chain.run(user_question)
            # Display the answer [cite: 24]
            st.write("**Evee's Answer:**")
            st.write(answer)