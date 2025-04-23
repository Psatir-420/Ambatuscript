import os
import json
import streamlit as st
from vector_store import VectorStore
from rag_engine import RAGEngine
import google.generativeai as genai

# Setup page config
st.set_page_config(
    page_title="Rechtwijs AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states if they don't exist
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pending_doc_request' not in st.session_state:
    st.session_state.pending_doc_request = None
if 'available_documents' not in st.session_state:
    st.session_state.available_documents = []

# Title and description
st.title("🇮🇩 Asisten AI berbahasa indonesia")
st.markdown("AI dengan basis data peraturan dan literatur berbahasa indonesia, By: Psatir-420(Gungde)")

# Sidebar for settings and actions
with st.sidebar:
    st.header("Settings")
    
    # API Key input
    api_key = st.text_input(
        "Pass Key", 
        value=st.session_state.gemini_api_key,
        type="password",
        help="not a member of tuweb ? Get your own pass key from https://aistudio.google.com/"
    )
    
    if api_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key
        
        # Reset RAG engine if API key changes
        st.session_state.rag_engine = None
        st.success("Password berhasil di submit")
    
    # Load data button
    if st.button("Load Data"):
        with st.spinner("Menunggu koneksi database"):
            try:
                # Initialize vector store
                data_dir = "processed_data"
                st.session_state.vector_store = VectorStore(data_dir)
                st.session_state.vector_store.load_documents()
                
                # Collect available document names
                st.session_state.available_documents = [
                    os.path.basename(doc["source"]) for doc in st.session_state.vector_store.documents
                ]
                
                # Show success message with document count
                doc_count = len(st.session_state.vector_store.documents)
                chunk_count = sum(len(doc["chunks"]) for doc in st.session_state.vector_store.documents)
                st.success(f"Successfully loaded {doc_count} documents with {chunk_count} chunks!")
                
                # Initialize RAG engine if we have a key
                if st.session_state.gemini_api_key:
                    st.session_state.rag_engine = RAGEngine(
                        st.session_state.vector_store,
                        st.session_state.gemini_api_key
                    )
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Show data stats if loaded
    if st.session_state.vector_store and st.session_state.vector_store.documents:
        st.metric("Documents Loaded", len(st.session_state.vector_store.documents))
        st.metric("Total Chunks", sum(len(doc["chunks"]) for doc in st.session_state.vector_store.documents))
        
        # Show list of available documents
        with st.expander("Available Documents"):
            for doc_name in st.session_state.available_documents:
                st.write(f"- {doc_name}")
    
    # Clear chat button
    if st.button("Mulai Percakapan Baru"):
        st.session_state.chat_history = []
        st.session_state.pending_doc_request = None
        st.success("Percakapan baru dimulai!")
    
    st.divider()
    st.markdown("### About")
    st.markdown("""
    Program ini hanyalah asisten semata, gunakan secukupnya dan dengan bertanggung jawab. Semua jawaban AI didasarkan pada data yang digunakan untuk melatih. Punya literatur hukum atau peraturan yang ingin anda tambahkan ? Chat langsung Gungde/ kirim PDF ke Grup Tuweb    
- Masukkan kunci akses Anda untuk menggunakan layanan ini
- Muat data hukum dari direktori data
- Ajukan pertanyaan terkait hukum Indonesia
- Dapatkan jawaban akurat dengan sumber kutipan
    """)

# Main content - Chat Interface
st.header("Hallo, Selamat Belajar")

# Check if we have required components
if not st.session_state.vector_store or len(st.session_state.vector_store.documents) == 0:
    st.warning("Masukkan kunci terlebih dahulu lalu load data untuk menggunakan AI")
elif not st.session_state.gemini_api_key:
    st.error("Masukkan kunci terlebih dahulu lalu load data untuk menggunakan AI")
elif not st.session_state.rag_engine:
    # Try to initialize RAG engine if vector store and API key are available
    try:
        st.session_state.rag_engine = RAGEngine(
            st.session_state.vector_store,
            st.session_state.gemini_api_key
        )
        st.success("RAG engine initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing RAG engine: {str(e)}")
else:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Show sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander("Lihat Sumber"):
                            for i, source in enumerate(message["sources"]):
                                st.write(f"**Sumber {i+1}:** {os.path.basename(source['source'])}")
                                st.write(f"**Halaman:** {source['metadata']['page_start']}-{source['metadata']['page_end']}")
                                st.text_area(f"Konten sumber {i+1}", 
                                          value=source['text'], 
                                          height=100,
                                          key=f"source_{message['id']}_{i}")
    
    # Handle document request if pending
    if st.session_state.pending_doc_request:
        st.info(f"AI meminta dokumen tambahan: {st.session_state.pending_doc_request}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Setuju"):
                # Add system message indicating approval
                st.session_state.chat_history.append({
                    "role": "system",
                    "content": f"User menyetujui permintaan dokumen: {st.session_state.pending_doc_request}",
                    "id": len(st.session_state.chat_history),
                    "visible": False  # Hidden from UI but available to RAG
                })
                st.session_state.pending_doc_request = None
                st.rerun()
        
        with col2:
            if st.button("Tolak"):
                # Add system message indicating rejection
                st.session_state.chat_history.append({
                    "role": "system",
                    "content": f"User menolak permintaan dokumen: {st.session_state.pending_doc_request}",
                    "id": len(st.session_state.chat_history),
                    "visible": False  # Hidden from UI but available to RAG
                })
                st.session_state.pending_doc_request = None
                st.rerun()
    
    # Settings for response
    with st.expander("Pengaturan Jawaban", expanded=False):
        num_results = st.number_input("Mau pakai berapa sumber ? Maximum 10 sumber sob(semakin banyak semakin lambat)", 
                                    min_value=1, 
                                    max_value=10, 
                                    value=3)
    
    # Input for new chat message
    user_input = st.chat_input("Ketik pertanyaan Anda tentang hukum Indonesia...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "id": len(st.session_state.chat_history)
        })
        
        with st.spinner("Lagi Mikir........"):
            # Get response from RAG engine with full chat history
            response = st.session_state.rag_engine.generate_response_with_chat(
                user_input, 
                st.session_state.chat_history,
                num_results=num_results,
                available_documents=st.session_state.available_documents
            )
            
            # Check if the response includes a document request
            if "document_request" in response and response["document_request"]:
                st.session_state.pending_doc_request = response["document_request"]
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
                "id": len(st.session_state.chat_history)
            })
        
        # Rerun the app to show the updated chat
        st.rerun()

# Footer
st.divider()
st.caption("Asisten AI Hukum Indonesia | By: Psatir-420 (Gungde)")
