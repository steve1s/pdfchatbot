import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.globals import set_llm_cache
from streamlit_chat import message
from langchain_community.embeddings import HuggingFaceEmbeddings


os.environ["OPENAI_API_KEY"] = "sk-abcd1234efgh5678abcd1234efgh5678abcd1234"

# --- Function to extract text from PDFs ---
def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# --- Function to split text into chunks ---
def get_text_chunks(text):
    """Splits a long text into smaller chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Creates a FAISS vector store from text chunks using free HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# --- Function to create a vector store from text chunks ---
def get_vectorstore(text_chunks):
    """Creates a FAISS vector store from text chunks using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# --- Function to create a conversation chain ---
def get_conversation_chain(vectorstore):
    """Creates a conversational retrieval chain."""
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# --- Main Streamlit App Logic ---
def main():
    load_dotenv()
    st.set_page_config(page_title="DocuChat: Chat with your PDFs", page_icon="📄")

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("DocuChat: Chat with your PDFs 📄💬")
    
    # User input form
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # --- Sidebar for PDF Upload and Processing ---
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    # 1. Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # 2. Get text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # 3. Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    
                    # 4. Create conversation chain and store in session state
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Processing complete! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

    # --- Display chat history ---
    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(msg.content, is_user=True, key=f"user_{i}")
            else:
                message(msg.content, is_user=False, key=f"bot_{i}")

def handle_userinput(user_question):
    """Handles user input and updates chat history."""
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
    else:
        st.warning("Please upload and process your documents first.")


if __name__ == '__main__':
    main()