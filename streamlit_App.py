from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory

import streamlit as st

st.set_page_config(page_title="AI Teaching Assistant", layout="wide")

st.markdown("""

### How It Works

Follow these simple steps to interact with the Mac AI Assistant:

1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")


import os
import time
current_directory = os.getcwd()

# Define function to save uploaded file
def save_uploaded_files(uploaded_files):
    saved_files = []

    user_files_directory = 'user_files'
    if not os.path.exists(user_files_directory):
        os.makedirs(user_files_directory)

    # Get the current working directory
    current_directory = os.getcwd()

    for uploaded_file in uploaded_files:
        file_path = os.path.join(current_directory,user_files_directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
    return saved_files

# Function to get last modified time of file
def get_modified_time(file):
  ti_m = os.path.getmtime(file)
  m_ti = time.ctime(ti_m)
  t_obj = time.strptime(m_ti)
  T_stamp = time.strftime("%Y-%m-%d %H:%M:%S", t_obj)
  return T_stamp

def replace_newlines(text):
    # Replace newline and carriage return + line feed characters with spaces
    return text.replace('\n', ' ').replace('\r\n', ' ').replace('\x0c', ' ')

def fix_missing_spaces(text):
    # Split text into sentences
    sentences = text.split('. ')
    # Add space after period for each sentence
    fixed_text = '. '.join(sentence + (' ' if i < len(sentences) - 1 else '') for i, sentence in enumerate(sentences))
    return fixed_text


from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# Function to extract text and metadata of pdf files
def prepare_docs(pdf_docs):
    docs = []
    metadata = []
    content = []

    for pdf in pdf_docs:
      for page_number, page_layout in enumerate(extract_pages(pdf), start=1):
        text = ""
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text += element.get_text()

        text = replace_newlines(text)
        text = fix_missing_spaces(text)
        doc_page = {'Title': pdf.split("\\")[-1] + " Page No: " + str(page_number),
                    'Last_modified_time': get_modified_time(pdf),
                    'Content': text,
                    'Source': "empty_url"}
        docs.append(doc_page)

    for doc in docs:
        content.append(doc["Content"])
        metadata.append({
            "Title": doc["Title"],
            "Last_modified_time": doc["Last_modified_time"],
            "Source": doc["Source"]
        })

    print("Content and metadata are extracted from the documents")

    return content, metadata


# split extracted text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_chunks(content, metadata):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512,chunk_overlap=15)
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Documents are split into {len(split_docs)} passages")
    return split_docs


# vector database
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_into_vectordb(split_docs):
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_docs, embedding_model)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    print("Vector database is created")
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever


# Langchain chain for sequence of call
def get_conversational_chain():

    prompt_template ="""
    You are a helpful Teaching Assistant of the McMaster University.\n
    This is the conversation between a student and and 'Mac AI Assistant". your job is to answer the student's question.\n
    The question can be a new question or follow up. So, you must check the chat histroy given below before you answer the question.\n
    You must answer student's the question based on only context given below.\n
    If the question can not be answered using the information provided in the context, must answer with I don't know, don't try to make up an answer.\n
    Give answers in natural form, without giving context as of what you're doing internally.\n
    Use three sentences maximum to answer a question. Keep the answer as concise as possible.\n
    If an answer to the question is provided uisng the context data, it must be annotated with a citation at the end of the answer. should use the following format to cite all relevant sources specified in the Sources. "\nCitation: \n1.source1.pdf Page No: xx  Source URL: xxx \n2.source2.pdf Page No: xx  Source URL: xxx \n3.source3.pdf Page No: xx  Source URL: xxx".\n 
    when citation is needed, must cite given resources in the Sources\n.
    If user question is more general for eaxmple 'Hi', 'Hi there!, 'Thanks', or 'How are you!', then asnwer them like a personal assistant of an user and do not need citation in such answers \n

    ""context:\n{context}\n\n""

    ""Chat history: \n{chat_history}\n\n""

    ""Student's Question: \n{question}\n\n""

    Answer: """

    # Langchain Prompt template to configure prompt variable
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question","chat_history"])
    model_path=os.path.join(current_directory, "llama-2-7b-chat.Q4_K_M.gguf")

    llama_llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=15000,
    n_threads=6,     
    n_batch=512,
    temperature=0.7,
    f16_kv=True,
    max_tokens=512,
    top_p=0.95,
    n_ctx=4000,
    verbose=False,
    streaming=False)

    # Memory to store chat history
    memory=ConversationBufferWindowMemory(k=2,memory_key="chat_history", return_messages=True, input_key="question")

    # Chain for sequence of call
    chain = load_qa_chain(llama_llm, chain_type="stuff", prompt=prompt, memory=memory,verbose=False)

    return chain


from langchain.docstore.document import Document

# Function add retrived chunk and metadata
def Add_text_with_metadata(docs):
    text=""
    data=""
    for doc in docs:
        text+=doc.page_content+"\n"
        data+=doc.metadata['Title']+" Source URL: "+ doc.metadata['Source']+"\n"
    final_text=text +"Sources: \n"+data
    doc =  Document(page_content=f"{final_text}")
    return [doc]

# Function for user input
def get_vector_store(text_chunks):
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(text_chunks, embedding_model)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)

def streamlite_user_input(user_question):
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    index = FAISS.load_local("vectorstore/db_faiss/",embedding_model,allow_dangerous_deserialization=True)
    retriever = index.as_retriever(search_kwargs={"k": 3})
    retrived_docs = retriever.get_relevant_documents(user_question)
    docs=Add_text_with_metadata(retrived_docs)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question})
    return response["output_text"]

def main():
    st.header("McMaster University | AI Teaching Assistant 💁")

    # Instructer panel
    with st.sidebar:
        st.title("McMaster Instructer Panel:")
        st.title(" ")
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button"):
            with st.spinner("Processing..."):
                pdf_docs=save_uploaded_files(pdf_docs)
                content, metadata = prepare_docs(pdf_docs)
                text_chunks = get_text_chunks(content, metadata)
                get_vector_store(text_chunks)
                st.success("Done")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
          st.markdown(message["content"])

    # Accept user input
    if user_question := st.chat_input("Ask a Question", key="user_question") :

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_question)

        # Display assistant response in chat message container
        with st.chat_message("Mac AI Assistant"):
            response = streamlite_user_input(user_question)
            st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "Mac AI Assistant", "content": response})

if __name__ == "__main__":
    main()

#App run Command
#streamlit run "C:\Users\rutvi\OneDrive\Desktop\NLP Project\Final_code\streamlit_App.py" [ARGUMENTS]