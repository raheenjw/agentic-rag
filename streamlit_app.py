import asyncio
import concurrent.futures
from collections.abc import Coroutine
from pathlib import Path
import time
from typing import TypeVar

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
from vector_db import QdrantStorage

load_dotenv()

st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

T = TypeVar("T")
_async_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def run_async(coro: Coroutine[object, object, T]) -> T:
    """Run async code off Streamlit's thread so httpx can close before the loop shuts down."""
    future = _async_executor.submit(asyncio.run, coro)
    return future.result()


def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


# Custom CSS for warm beige/brown theme
st.markdown("""
<style>
    /* FORCE beige background on EVERYTHING */
    * {
        background-color: transparent !important;
    }
    
    html, body, .stApp, .main, .block-container, 
    [data-testid="stAppViewContainer"], [data-testid="stApp"],
    .appview-container {
        background-color: #f5f1e8 !important;
    }
    
    /* Main content area */
    .main .block-container, .main, section.main {
        background-color: #f5f1e8 !important;
        padding-top: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div,
    .css-1d391kg, aside {
        background-color: #e8dcc8 !important;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #5c4a3a;
        margin-bottom: 0.5rem;
        background-color: transparent !important;
    }
    .sub-header {
        font-size: 1rem;
        color: #8b7355;
        margin-bottom: 2rem;
        background-color: transparent !important;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #6b5444;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #d4c4b0;
        padding-bottom: 0.5rem;
        background-color: transparent !important;
    }
    
    /* Text colors */
    p, span, label, div, h1, h2, h3, h4, h5, h6 {
        color: #4a3f35 !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #c9b99b !important;
        color: #4a3f35 !important;
        border: none !important;
        border-radius: 0.5rem;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #b8a788 !important;
    }
    
    /* Primary buttons */
    .stButton button[kind="primary"], button[data-baseweb="button"][kind="primary"] {
        background-color: #8b7355 !important;
        color: #ffffff !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #6b5444 !important;
    }
    
    /* Secondary buttons */
    .stButton button[kind="secondary"] {
        background-color: #d4c4b0 !important;
        color: #4a3f35 !important;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"], [data-testid="stChatMessageContent"],
    .stChatMessage {
        background-color: #faf7f2 !important;
        border-radius: 0.75rem;
        border: 1px solid #e8dcc8;
    }
    
    /* Chat input - COMPREHENSIVE */
    [data-testid="stChatInput"], [data-testid="stChatInput"] > div,
    [data-testid="stChatInputTextArea"], .stChatInput,
    .stChatFloatingInputContainer, [data-testid="stBottom"] {
        background-color: #f5f1e8 !important;
    }
    
    [data-testid="stChatInput"] textarea, 
    [data-testid="stChatInput"] input {
        background-color: #faf7f2 !important;
        color: #4a3f35 !important;
        border: 1px solid #d4c4b0 !important;
        border-radius: 0.5rem;
    }
    
    /* All input fields */
    input, textarea, select {
        background-color: #faf7f2 !important;
        color: #4a3f35 !important;
        border: 1px solid #d4c4b0 !important;
        border-radius: 0.5rem;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"], [data-testid="stFileUploader"] *,
    .stFileUploader {
        background-color: #faf7f2 !important;
        border: 2px dashed #d4c4b0;
        border-radius: 0.5rem;
    }
    
    /* Metrics */
    [data-testid="stMetric"], [data-testid="stMetricValue"], 
    [data-testid="stMetricLabel"] {
        background-color: #faf7f2 !important;
        color: #5c4a3a !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e8dcc8;
    }
    
    /* Expanders */
    .streamlit-expanderHeader, details, summary {
        background-color: #faf7f2 !important;
        color: #6b5444 !important;
        border-radius: 0.5rem;
        border: 1px solid #e8dcc8;
    }
    
    /* Messages */
    .stSuccess, [data-testid="stSuccess"] {
        background-color: #d4e5d4 !important;
        color: #2e5d2e !important;
        border-left: 4px solid #6b9b6b;
        border-radius: 0.5rem;
    }
    
    .stInfo, [data-testid="stInfo"] {
        background-color: #d4dce8 !important;
        color: #2e4d6b !important;
        border-left: 4px solid #5a7fa8;
        border-radius: 0.5rem;
    }
    
    .stError, [data-testid="stError"] {
        background-color: #e8d4d4 !important;
        color: #6b2e2e !important;
        border-left: 4px solid #a85a5a;
        border-radius: 0.5rem;
    }
    
    /* Spinner */
    .stSpinner > div, [data-testid="stSpinner"] {
        border-top-color: #8b7355 !important;
    }
    
    /* Captions */
    .stCaption, [data-testid="stCaption"] {
        color: #8b7355 !important;
        background-color: transparent !important;
    }
    
    /* Divider */
    hr {
        border-color: #d4c4b0 !important;
        margin: 1.5rem 0;
        background-color: transparent !important;
    }
    
    /* Bottom area and footer */
    footer, [data-testid="stFooter"], .stChatFloatingInputContainer {
        background-color: #f5f1e8 !important;
    }
    
    /* Override any remaining white */
    div[class*="st"], section[class*="st"], 
    div[data-baseweb], section[data-baseweb] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Chat </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent document analysis powered by AI agents</div>', unsafe_allow_html=True)

# Initialize chat sessions structure FIRST
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = str(int(time.time() * 1000))
    st.session_state.chat_sessions[st.session_state.current_thread_id] = {
        "title": "New Conversation",
        "timestamp": time.time(),
        "messages": []
    }

# Get current chat
current_chat = st.session_state.chat_sessions.get(st.session_state.current_thread_id, {"messages": []})
if "messages" not in st.session_state:
    st.session_state.messages = current_chat.get("messages", [])

# Sidebar - Document Management
with st.sidebar:
    st.markdown("### Knowledge Base")
    st.caption("Upload documents to add them to the knowledge base")
    
    # Track processed files to avoid re-uploading
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    
    # Document upload in sidebar
    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="pdf_uploader"
    )
    
    if uploaded is not None:
        # Create a unique identifier for this file
        file_id = f"{uploaded.name}_{uploaded.size}"
        
        if file_id not in st.session_state.processed_files:
            with st.spinner("Indexing document..."):
                path = save_uploaded_pdf(uploaded)
                run_async(send_rag_ingest_event(path))
                time.sleep(0.3)
                st.session_state.processed_files.add(file_id)
            st.success(f"Added: {path.name}")
        else:
            st.info(f"Already indexed: {uploaded.name}")
    
    st.divider()
    
    # Display indexed documents
    try:
        storage = QdrantStorage()
        sources = storage.list_sources()
        if sources:
            st.caption(f"**{len(sources)} document(s) indexed**")
            
            for src in sources:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(src)
                with col2:
                    if st.button("×", key=f"del_{src}", help=f"Remove {src}"):
                        storage.delete_by_source(src)
                        st.rerun()
            
            st.divider()
            if st.button("Clear All", type="secondary", use_container_width=True):
                if storage.clear_all():
                    st.success("Knowledge base cleared")
                    st.rerun()
                else:
                    st.error("Operation failed")
        else:
            st.info("No documents indexed")
    except Exception as e:
        st.error(f"Error: {e}")

# Main content area
st.markdown('<div class="section-header">AI Assistant</div>', unsafe_allow_html=True)
st.caption("Ask questions about your documents. The assistant searches across all indexed documents.")

# Sidebar - Chat History
with st.sidebar:
    st.divider()
    st.markdown("### Conversations")
    
    # New Chat button
    if st.button("+ New Conversation", use_container_width=True, type="primary"):
        # Save current chat before creating new one
        if st.session_state.messages:
            st.session_state.chat_sessions[st.session_state.current_thread_id]["messages"] = st.session_state.messages
        
        # Create new chat
        new_thread_id = str(int(time.time() * 1000))
        st.session_state.current_thread_id = new_thread_id
        st.session_state.chat_sessions[new_thread_id] = {
            "title": "New Conversation",
            "timestamp": time.time(),
            "messages": []
        }
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Display chat sessions (most recent first)
    sorted_sessions = sorted(
        st.session_state.chat_sessions.items(),
        key=lambda x: x[1]["timestamp"],
        reverse=True
    )
    
    for thread_id, session in sorted_sessions:
        is_current = thread_id == st.session_state.current_thread_id
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            # Generate title from first message if available
            title = session.get("title", "New Conversation")
            if session.get("messages") and title == "New Conversation":
                first_msg = session["messages"][0]["content"][:35]
                title = f"{first_msg}..." if len(session["messages"][0]["content"]) > 35 else first_msg
                session["title"] = title
            
            # Format timestamp
            timestamp = session.get("timestamp", time.time())
            time_str = time.strftime("%H:%M", time.localtime(timestamp))
            
            button_label = f"{'▸ ' if is_current else ''}{title}"
            button_type = "primary" if is_current else "secondary"
            
            if st.button(
                button_label,
                key=f"chat_{thread_id}",
                use_container_width=True,
                disabled=is_current,
                help=f"Started at {time_str}",
                type=button_type
            ):
                # Save current chat
                st.session_state.chat_sessions[st.session_state.current_thread_id]["messages"] = st.session_state.messages
                
                # Switch to selected chat
                st.session_state.current_thread_id = thread_id
                st.session_state.messages = session["messages"].copy()
                st.rerun()
        
        with col2:
            if st.button("×", key=f"del_chat_{thread_id}", help="Delete conversation"):
                del st.session_state.chat_sessions[thread_id]
                
                # If deleted current chat, switch to most recent or create new
                if thread_id == st.session_state.current_thread_id:
                    if st.session_state.chat_sessions:
                        # Switch to most recent chat
                        st.session_state.current_thread_id = sorted_sessions[0][0] if sorted_sessions[0][0] != thread_id else (sorted_sessions[1][0] if len(sorted_sessions) > 1 else None)
                        if st.session_state.current_thread_id:
                            st.session_state.messages = st.session_state.chat_sessions[st.session_state.current_thread_id]["messages"].copy()
                        else:
                            # Create new chat if none left
                            new_thread_id = str(int(time.time() * 1000))
                            st.session_state.current_thread_id = new_thread_id
                            st.session_state.chat_sessions[new_thread_id] = {
                                "title": "New Conversation",
                                "timestamp": time.time(),
                                "messages": []
                            }
                            st.session_state.messages = []
                    else:
                        # Create new chat if none left
                        new_thread_id = str(int(time.time() * 1000))
                        st.session_state.current_thread_id = new_thread_id
                        st.session_state.chat_sessions[new_thread_id] = {
                            "title": "New Conversation",
                            "timestamp": time.time(),
                            "messages": []
                        }
                        st.session_state.messages = []
                st.rerun()
    
    st.caption(f"Session ID: {st.session_state.current_thread_id[:12]}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for src in message["sources"]:
                    st.text(f"• {src}")
        if message.get("chunks"):
            with st.expander(f"Retrieved Context ({len(message['chunks'])} chunks)"):
                for i, chunk in enumerate(message["chunks"], 1):
                    st.markdown(f"**Chunk {i}** — *{chunk['source']}*")
                    st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
                    if i < len(message["chunks"]):
                        st.divider()


async def send_rag_query_event(question: str, thread_id: str) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": 5,
                "thread_id": thread_id,
            },
        )
    )
    return result[0]


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


# Chat input
if prompt := st.chat_input("Ask a question about your PDFs..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Save to chat session
    st.session_state.chat_sessions[st.session_state.current_thread_id]["messages"] = st.session_state.messages
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing query..."):
            try:
                event_id = run_async(send_rag_query_event(prompt, st.session_state.current_thread_id))
                output = wait_for_run_output(event_id)
                answer = output.get("answer", "Unable to generate response.")
                sources = output.get("sources", [])
                chunks = output.get("chunks", [])
                
                st.markdown(answer)
                
                if sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        for src in sources:
                            st.text(f"• {src}")
                
                if chunks:
                    with st.expander(f"Retrieved Context ({len(chunks)} chunks)"):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Chunk {i}** — *{chunk['source']}*")
                            st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
                            if i < len(chunks):
                                st.divider()
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "chunks": chunks
                })
                
                # Save to chat session
                st.session_state.chat_sessions[st.session_state.current_thread_id]["messages"] = st.session_state.messages
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "chunks": []
                })
                
                # Save to chat session
                st.session_state.chat_sessions[st.session_state.current_thread_id]["messages"] = st.session_state.messages

