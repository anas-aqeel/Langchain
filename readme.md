# Majic Chat: AI-Powered Document Assistant

## Overview

Majic Chat is an AI-powered document assistant that allows users to upload documents (e.g., PDFs) and ask questions about them. The app uses OpenAI embeddings to process the document, create vector stores, and retrieve relevant document sections in response to user queries. It is built using Streamlit for the UI and LangChain for document processing and retrieval.

The project also includes a pure Python version (`program.ipynb`) of the same functionality for environments without Streamlit.

---

## Folder Structure

- **assets/**
  - `logo.jpeg` : The project logo used in the UI.
  
- **.env.example**: Template for environment variables like API keys.

- **index.py**: Main Streamlit app.

- **program.ipynb**: Pure Python implementation of Majic Chat's document retrieval functionality without the Streamlit UI.

- **readme.md**: This documentation file.

- **requirements.txt**: List of required Python packages.

---

## Requirements

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

### API Token

Ensure you have your Azure OpenAI API key and endpoint set in a `.env` file similar to `.env.example`.

---

## Azure OpenAI Token Limits

When using Azure OpenAI embeddings, note the following limits:
- **Maximum token size**: 64,000 tokens/request (embedding model).
- **Request rate limit**: 15 requests per minute.
- **Daily request limit**: 150 requests/day.

Additionally, for the `gpt-4o-mini` model (used for query responses):
- **Max tokens per request**: 8,000 in, 4,000 out (as shown in the image).
- **Concurrent requests**: 5.

---

## Program Flow

### Streamlit App (`index.py`)

1. **Initialization**
   - The environment variables are loaded from `.env`.
   - The session state variables for storing chats and vector stores are initialized.

2. **Upload Document**
   - Users can upload a document (PDF format supported).
   - The app splits the document into chunks, embeds them using Azure OpenAI, and saves the resulting vector store to a file.
   - If the document exceeds 100 pages, the app will display an error.

3. **Chat Interaction**
   - Users can interact with the chat interface to ask questions based on the uploaded document.
   - The app retrieves relevant document chunks using FAISS and uses OpenAI's GPT to generate a comprehensive answer.

4. **Save and Load Chats**
   - Chat histories and vector store paths are saved to `chats.json` and loaded at the start of the app.

### Pure Python Version (`program.ipynb`)

1. **Load Document**
   - Similar to the Streamlit version, documents are loaded, split, and embedded using OpenAI embeddings.

2. **Question Answering**
   - The Python version implements a command-line interface for loading documents and querying them without using Streamlit.

---

## Key Functions

- **`create_new_chat()`**: Creates a new chat session and stores it in the session state.
  
- **`save_chats()`**: Saves chat history and vector store metadata (not the vector store itself) to `chats.json`.

- **`save_document_vectorstore()`**: Saves the FAISS vector store for a specific chat to a file.

- **`load_document_vectorstore()`**: Loads an existing vector store from a file based on the chat ID.

- **`process_document()`**: Processes the uploaded document (splitting, embedding, saving the vector store).

- **`get_llm_response()`**: Retrieves a response from OpenAI's GPT model using the vector store's retrieved chunks as context.

---

## Usage

### Streamlit Version
To run the Streamlit app:

```bash
streamlit run index.py
```

Upload a document and start interacting with Majic Chat!

### Pure Python Version
For the command-line version, open and run `program.ipynb` in a Python environment, following the provided steps for document loading and querying.

---

## License
This project is licensed under the MIT License.

---
