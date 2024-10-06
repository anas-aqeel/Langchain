import streamlit as st
import tempfile
import os
import json
import uuid
from langchain_community.document_loaders import CSVLoader, JSONLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import PyPDF2
from langchain_openai import ChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# loading dot env
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Majic Chat", layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS for sidebar
st.markdown("""
    <style>
    .sidebar {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'chats' not in st.session_state:
    st.session_state.chats = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'vectorstores' not in st.session_state:
    st.session_state.vectorstores = {}

# Azure OpenAI configuration
TOKEN = os.getenv("GITHUB_TOKEN")
ENDPOINT = "https://models.inference.ai.azure.com"
EMBEDDINGS_MODEL_NAME = "text-embedding-3-large"
LLM_MODEL_NAME = "gpt-4o-mini"

embeddings = AzureOpenAIEmbeddings(
    model=EMBEDDINGS_MODEL_NAME,
    azure_endpoint=ENDPOINT,
    api_key=TOKEN
)


def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": f"Chat {len(st.session_state.chats) + 1}",
        "history": []
    }
    st.session_state.current_chat_id = chat_id
    st.query_params["chat"] = chat_id
    return chat_id


def save_chats():
    serializable_chats = {
        chat_id: {
            "title": chat_data["title"],
            "history": [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "context": msg.get("context", [])
                } for msg in chat_data["history"]
            ],
            "vectorstore_path": chat_data["vectorstore_path"] if "vectorstore_path" in chat_data else None
        } for chat_id, chat_data in st.session_state.chats.items()
    }
    with open('chats.json', 'w') as f:
        json.dump(serializable_chats, f)


def save_document_vectorstore(chat_id, vectorstore):
    # Save the vectorstore separately (e.g., to a file or different storage)
    # For example, you can use FAISS's built-in save/load methods to handle this
    vectorstore.save_local(f'vectorstore_{chat_id}.faiss')

    # Store only the file path or metadata in the JSON (not the actual vectorstore)
    st.session_state.chats[chat_id]["vectorstore_path"] = f'vectorstore_{
        chat_id}.faiss'

    # Save chats to the JSON file without vectorstore object
    with open('chats.json', 'w') as f:
        json.dump(st.session_state.chats, f)


def load_chats():
    if os.path.exists('chats.json'):
        with open('chats.json', 'r') as f:
            st.session_state.chats = json.load(f)


def load_document_vectorstore(chat_id):
    # Load the vectorstore from the saved file path
    if os.path.exists('chats.json'):
        with open('chats.json', 'r') as f:
            chats = json.load(f)
            if chat_id in chats:
                vectorstore_path = chats[chat_id].get("vectorstore_path")
                if vectorstore_path and os.path.exists(vectorstore_path):
                    vectorstore = FAISS.load_local(
                        vectorstore_path, embeddings, allow_dangerous_deserialization=True)
                    st.session_state.vectorstores[chat_id] = vectorstore
                    return vectorstore


def process_document(file, chat_id):
    file_extension = file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    try:
        if file_extension == 'pdf':
            with open(temp_file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                if len(pdf_reader.pages) > 100:
                    st.error(
                        "⚠️ PDF exceeds 100 pages. Please upload a smaller PDF.")
                    return  # Avoid triggering st.rerun() if PDF is too large
                else:
                    loader = PyPDFLoader(file_path=temp_file_path)
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=3000, chunk_overlap=500)
                    chunks = text_splitter.split_documents(data)

                    st.session_state.vectorstores[chat_id] = FAISS.from_documents(
                        documents=chunks, embedding=embeddings)
                    save_document_vectorstore(
                        chat_id, st.session_state.vectorstores[chat_id])
                    st.rerun()
        else:
            st.error(
                "⚠️ Unsupported file type. Please upload a PDF file.")
            return

    finally:
        os.remove(temp_file_path)


def get_llm_response(user_question, vectorstore):
    llm = ChatOpenAI(
        openai_api_key=TOKEN,
        openai_api_base=ENDPOINT,
        model_name=LLM_MODEL_NAME,
        temperature=1.0,
        max_tokens=4000
    )

    system_prompt = """
    Use the following context to either answer the user's question or explain their input text, depending on what they provide:
    - If the user asks a question, answer it comprehensively by incorporating the context and adding extra value or clarity where relevant.
    - If the user does not ask a question but provides a statement or piece of text, explain it in relation to the given context.

    <context>
    {context}
    </context>

    User's Query: {question}
    """

    qa_prompt = ChatPromptTemplate.from_template(system_prompt)
    question_answer_chain = create_stuff_documents_chain(
        llm=llm, prompt=qa_prompt)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_chunks = retriever.get_relevant_documents(user_question)

    result = question_answer_chain.invoke({
        "question": user_question,
        "context": relevant_chunks
    })

    return result, [chunk.page_content for chunk in relevant_chunks]


# Load existing chats
load_chats()


# Sidebar
with st.sidebar:
    st.image("assets/logo.jpeg", width=100)
    st.title("Majic Chat")
    st.subheader("The RAG Chat")

    if st.button("🆕 New Chat", key="new_chat_sidebar"):
        new_chat_id = create_new_chat()
        save_chats()
        st.rerun()

    st.markdown("---")
    st.subheader("Your Chats")
    for chat_id, chat_data in st.session_state.chats.items():
        if st.button(f"💬 {chat_data['title'][:30]}...", key=chat_id):
            st.session_state.current_chat_id = chat_id
            st.query_params["chat"] = chat_id
            st.rerun()

    st.markdown("---")
    if st.button("🗑 Clear All Chats", key="clear_chats"):
        st.session_state.chats = {}
        st.session_state.current_chat_id = None
        st.session_state.vectorstores = {}
        save_chats()
        st.query_params.clear()
        st.rerun()

    if st.button("🆕 New Chat", key="new_chat_bottom"):
        new_chat_id = create_new_chat()
        save_chats()
        st.rerun()

# Main content
st.markdown("<div class='centered'>", unsafe_allow_html=True)
st.image("assets/logo.jpeg", width=200)
st.title("Majic Chat: Your AI-Powered Document Assistant")
st.markdown("</div>", unsafe_allow_html=True)

# Get the chat ID from the URL parameter (if any)
params = st.query_params
chat_id = params["chat"] if "chat" in params else None

# If a chat ID exists in the URL and is valid, use it
if chat_id and chat_id in st.session_state.chats:
    st.session_state.current_chat_id = chat_id
    # Load existing document vecotrstore
    load_document_vectorstore(st.session_state.current_chat_id)

# Check if the current chat ID exists and is valid
if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
    current_chat = st.session_state.chats[st.session_state.current_chat_id]
else:
    # If no valid chat is found, create a new one
    new_chat_id = create_new_chat()
    current_chat = st.session_state.chats[new_chat_id]

# Display chat title
st.subheader(f"Chat: {current_chat['title']}")

# File upload section
uploaded_file = st.file_uploader(
    "📁 Choose a file (PDF: max 20MB)", type=['pdf'])

# If a file is uploaded and there is no vectorstore for the current chat, process the document
if uploaded_file and st.session_state.current_chat_id not in st.session_state.vectorstores:
    process_document(uploaded_file, st.session_state.current_chat_id)

# Create a container for chat messages
chat_container = st.container()

# Display chat history with user and assistant messages
with chat_container:
    for message in current_chat['history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        if message['context']:
            with st.expander("Show Context", expanded=False):
                st.write(message["context"])


# Create a container for the input field and send button
input_container = st.container()

# Add the input field and send button to the container
with input_container:

    if user_input := st.chat_input(placeholder="Enter your question here"):
        if user_input:
            if st.session_state.current_chat_id in st.session_state.vectorstores:
                vectorstore = st.session_state.vectorstores[st.session_state.current_chat_id]

                # Store the user message and assistant response in the chat history
                current_chat['history'].append(
                    {"role": "user", "content": user_input})

                # Get response from LLM and append to chat history
                try:
                    with st.spinner("🤔 Thinking..."):
                        result, context = get_llm_response(
                            user_input, st.session_state.vectorstores[st.session_state.current_chat_id])
                        current_chat['history'].append({
                            "role": "assistant",
                            "content": result,
                            "context": context
                        })
                except Exception as e:
                    st.error(f"Error retrieving LLM response: {e}")

                # Save the updated chat history
                save_chats()
                st.rerun()
            else:
                st.error("No vectorstore found. Please upload a document.")


# Save chats on session end
if st.session_state.current_chat_id:
    save_chats()
