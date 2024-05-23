import os
import json
import chainlit as cl
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

from PyPDF2 import PdfReader
from typing import List

from chainlit.input_widget import TextInput

# Configuration
embedding_model_hf = "sentence-transformers/all-mpnet-base-v2"
index_OL_path = "data/vectorstore/temp-index.faiss"
index_mpnet_path = "data/vectorstore/index_mpnet.faiss"

model = Ollama(base_url="http://localhost:11434", model="llama3:instruct")

embeddings_HF = HuggingFaceEmbeddings(model_name=embedding_model_hf)
embeddings_OL = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text",
    show_progress="true",
    temperature=0,
)

# on décide ici quel index et quel modèle utiliser
embeddings = embeddings_HF
index_path = index_mpnet_path
faiss_index = None


def load_new_documents(directory):
    documents = load_documents_from_directory(directory)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=150
    )

    chunks = []
    for doc in documents:
        splits = text_splitter.create_documents(
            text_splitter.split_text(doc["content"]))
        for split in splits:
            split.metadata = {
                "source": doc["source"], **doc.get("metadata", {})}
            chunks.append(split)

    return chunks


def read_text_from_file(file_path: str) -> str:
    """Function to read PDF and return text"""

    if file_path.lower().endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            metadata = reader.metadata
            text = "\n".join(page.extract_text()
                             or "" for page in reader.pages)
            return text, metadata
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), {}
    elif file_path.lower().endswith(".json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.dumps(json.load(f)), {}
    else:
        raise ValueError(
            "Unsupported file type. Please upload a .txt or .pdf file.")

# Function to load documents individually


def add_documents_to_index(vectorstore, new_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=150)
    new_docs = []

    # Assurez-vous que new_documents est une liste de contenu de document
    for doc in new_documents:
        chunks = text_splitter.create_documents(doc.page_content)
        new_docs.extend(chunks)

    vectorstore.add_documents(new_docs)
    vectorstore.save_local(index_path)


def load_documents_from_directory(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith((".txt", ".pdf", ".json")):
                print("Traitement de ", filename)
                file_path = os.path.join(root, filename)
                try:
                    text, metadata = read_text_from_file(file_path)
                    documents.append(
                        {"content": text, "source": file_path, "metadata": metadata})
                except ValueError as e:
                    print(f"Error processing {file_path}: {e}")
    return documents


def add_documents(directory: str):
    retriever = cl.user_session.get("retriever")
    vectorstore = retriever.vectorstore

    new_documents = load_new_documents(directory)
    add_documents_to_index(vectorstore, new_documents)

    vectorstore.save_local(index_path)
    print("Nouveaux documents ajoutés et index mis à jour.")
