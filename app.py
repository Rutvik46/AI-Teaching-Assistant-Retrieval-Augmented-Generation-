# app.py
"""
McMaster AI Teaching Assistant
--------------------------------
Streamlit-based conversational app using LangChain + LlamaCpp + FAISS.
"""

import os
import time
import streamlit as st
from typing import List, Tuple

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp


# ---------------------------- CONFIG -------------------------------- #

APP_TITLE = "McMaster University | AI Teaching Assistant 💁"
USER_FILES_DIR = "user_files"
VECTORSTORE_PATH = "vectorstore/db_faiss"
MODEL_PATH = os.path.join(os.getcwd(), "llama-2-7b-chat.Q4_K_M.gguf")

# -------------------------------------------------------------------- #


# ---------------------- FILE HANDLING UTILS ------------------------- #
def save_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded PDF files locally and return file paths."""
    os.makedirs(USER_FILES_DIR, exist_ok=True)
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(USER_FILES_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
    return saved_files


def get_modified_time(file: str) -> str:
    """Return last modified timestamp of a file."""
    ti_m = os.path.getmtime(file)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ti_m))


# --------------------- PDF PROCESSING UTILS ------------------------- #
def clean_text(text: str) -> str:
    """Normalize extracted PDF text."""
    text = text.replace("\n", " ").replace("\r\n", " ").replace("\x0c", " ")
    return " ".join(text.split())  # remove extra spaces


def prepare_docs(pdf_paths: List[str]) -> Tuple[List[str], List[dict]]:
    """Extract content and metadata from PDF documents."""
    docs, metadata, content = [], [], []

    for pdf in pdf_paths:
        for page_number, page_layout in enumerate(extract_pages(pdf), start=1):
            text = "".join(
                element.get_text() for element in page_layout if isinstance(element, LTTextContainer)
            )
            text = clean_text(text)

            docs.append({
                "Title": f"{os.path.basename(pdf)} Page {page_number}",
                "Last_modified_time": get_modified_time(pdf),
                "Content": text,
                "Source": "empty_url"
            })

    for doc in docs:
        content.append(doc["Content"])
        metadata.append({
            "Title": doc["Title"],
            "Last_modified_time": doc["Last_modified_time"],
            "Source": doc["Source"],
        })

    return content, metadata


def split_text(content: List[str], metadata: List[dict]):
    """Split extracted content into chunks."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=15
    )
    return splitter.create_documents(content, metadatas=metadata)


# ------------------------ VECTORSTORE UTILS ------------------------- #
def build_vectorstore(docs):
    """Create FAISS vectorstore from docs."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTORSTORE_PATH)
    return db.as_retriever(search_kwargs={"k": 3})


def load_vectorstore():
    """Load FAISS vectorstore if exists."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)


# -------------------------- LLM + CHAIN ----------------------------- #
def get_conversational_chain():
    """Return conversational QA chain with memory."""
    prompt_template = """
    You are Mac AI Assistant, a helpful Teaching Assistant at McMaster University. Follow these guidelines when answering:\n\n
    1. Context-based answers: Use only the context provided. Do not mention the context explicitly in your answer.\n
    2. Honesty: If the answer cannot be found in the given context, respond with: "I don't know." Do not invent information.\n
    3. Style:\n
        - Keep answers concise (maximum three sentences).\n
        - Use a natural, conversational tone.\n
        - Always start your response with this format: Mac AI Assistant: <put your answer here> \n
    4. General interactions: If the user greets you or makes casual remarks (e.g., “Hi”, “Thanks”, “How are you”), respond like a friendly personal assistant.\n
    5. Answer in max 3 sentences. Add citations if applicable.

    Context:
    {context}

    Chat history:
    {chat_history}

    Student's Question:
    {question}

    
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    llama_llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=15000,
        n_threads=6,
        n_batch=512,
        temperature=0.7,
        max_tokens=512,
        top_p=0.95,
        n_ctx=4000,
        verbose=False,
        streaming=False,
    )

    memory = ConversationBufferWindowMemory(
        k=2, memory_key="chat_history", return_messages=True, input_key="question"
    )

    return load_qa_chain(llama_llm, chain_type="stuff", prompt=prompt, memory=memory, verbose=False)


def wrap_docs_with_metadata(docs):
    """Wrap retrieved documents with metadata for QA chain."""
    text, meta = "", ""
    for doc in docs:
        text += doc.page_content + "\n"
        meta += f"{doc.metadata['Title']} Source URL: {doc.metadata['Source']}\n"
    return [Document(page_content=f"{text}\nSources:\n{meta}")]


def query_chain(user_question: str):
    """Retrieve documents and query conversational chain."""
    index = load_vectorstore()
    retriever = index.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(user_question)
    docs = wrap_docs_with_metadata(retrieved_docs)

    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question})
    return response["output_text"]


# ---------------------------- STREAMLIT ------------------------------ #
def main():
    st.set_page_config(page_title="AI Teaching Assistant", layout="wide")

    # Sidebar - Document Upload
    with st.sidebar:
        st.title("📚 Instructor Panel")
        uploaded_files = st.file_uploader(
            "Upload PDF files", type=["pdf"], accept_multiple_files=True
        )

        if st.button("Process Documents"):
            with st.spinner("Processing..."):
                pdf_paths = save_uploaded_files(uploaded_files)
                content, metadata = prepare_docs(pdf_paths)
                docs = split_text(content, metadata)
                build_vectorstore(docs)
                st.success("Documents processed and indexed ✅")

        st.markdown("---")
        st.subheader("💬 Conversations")
        if "conversations" not in st.session_state:
            st.session_state.conversations = {}
        if "current_conv" not in st.session_state:
            st.session_state.current_conv = "Default"

        conv_names = list(st.session_state.conversations.keys()) or ["Default"]
        selected_conv = st.selectbox("Select a conversation", conv_names, index=0)
        st.session_state.current_conv = selected_conv

        if st.button("➕ New Conversation"):
            new_name = f"Conversation {len(st.session_state.conversations) + 1}"
            st.session_state.conversations[new_name] = []
            st.session_state.current_conv = new_name

    # Main Chat Window
    st.header(APP_TITLE)

    if st.session_state.current_conv not in st.session_state.conversations:
        st.session_state.conversations[st.session_state.current_conv] = []

    messages = st.session_state.conversations[st.session_state.current_conv]

    # Display past messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    if user_question := st.chat_input("Ask a question..."):
        messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            response = query_chain(user_question)
            st.markdown(response)
            messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
