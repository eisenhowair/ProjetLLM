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


from embedding_models import *


model = Ollama(base_url="http://localhost:11434", model="llama3:instruct")

embeddings_HF = HuggingFaceEmbeddings(model_name=embedding_model_hf_en_instructor_large)
embeddings_OL = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model=embedding_model_ol_en_nomic,
    show_progress="true",
    temperature=0,
)

# on décide ici quel index et quel modèle utiliser
embeddings = embeddings_HF
index_path = index_en_path_instructor_large
faiss_index = None


def load_new_documents(directory):
    documents = load_documents_from_directory(directory)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=150)

    chunks = []
    for doc in documents:
        splits = text_splitter.create_documents(
            text_splitter.split_text(doc["content"])
        )
        for split in splits:
            split.metadata = {"source": doc["source"], **doc.get("metadata", {})}
            chunks.append(split)

    return chunks


def change_model(new_model):
    # on change de modèle d'embedding pour en prendre un adapté à la langue

    if new_model == "instructor-large":
        embeddings_HF = HuggingFaceEmbeddings(
            model_name=embedding_model_hf_en_instructor_large
        )
        index_path = index_en_path_instructor_large

    elif new_model == "instructor-xl":
        embeddings_HF = HuggingFaceEmbeddings(
            model_name=embedding_model_hf_en_instructor_xl
        )
        index_path = index_en_path_instructor_xl

    elif new_model == "instructor-base":
        embeddings_HF = HuggingFaceEmbeddings(
            model_name=embedding_model_hf_en_instructor_base
        )
        index_path = index_en_path_instructor_base

    elif new_model == "mpnet-v2":
        embeddings_HF = HuggingFaceEmbeddings(model_name=embedding_model_hf_en_mpnet)
        index_path = index_en_path_mpnet

    elif new_model == "camembert-base":
        embeddings_HF = HuggingFaceEmbeddings(model_name=embedding_model_hf_fr)
        index_path = index_fr_path_camembert

    return embeddings_HF, index_path


def read_text_from_file(file_path: str) -> str:
    """Function to read PDF and return text"""

    if file_path.lower().endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            metadata = reader.metadata
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text, metadata
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), {}
    elif file_path.lower().endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.dumps(json.load(f)), {}
    else:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")


# Function to load documents individually


def add_documents_to_index(vectorstore, new_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=150)
    new_docs = []

    for doc in new_documents:
        chunks = text_splitter.create_documents(doc)
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
                        {"content": text, "source": file_path, "metadata": metadata}
                    )
                except ValueError as e:
                    print(f"Error processing {file_path}: {e}")
    return documents


def add_documents(directory: str):
    retriever = cl.user_session.get("retriever")
    vectorstore = retriever.vectorstore

    add_documents_to_index(vectorstore, directory)

    vectorstore.save_local(index_path)
    print("Nouveaux documents ajoutés et index mis à jour.")
