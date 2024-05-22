import os
import json
import chainlit as cl
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory

from operator import itemgetter
from PyPDF2 import PdfReader
from typing import List

# Configuration
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
index_path = "data/vectorstore/temp-index.faiss"

model = Ollama(base_url="http://localhost:11434", model="llama3:instruct")

# embeddings_HF = HuggingFaceEmbeddings(model_name=embedding_model)
embeddings_OL = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text",
    show_progress="true",
    temperature=0,
)

# Global index variable
faiss_index = None


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


@cl.step(type="run", name="Mise en place du Runnable")
def setup_model():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    prompt_exercice = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ton rôle est de répondre en francais à cette question {question} de l'utilisateur en te basant **uniquement** sur le contexte fourni. 
                Si tu ne trouves pas la réponse dans le contexte, demande à l'utilisateur d'être plus précis au lieu de deviner. 
                Voici le contexte nécessaire avec les sources :
                {context}""",
            ),
            MessagesPlaceholder(variable_name="history"),
        ]
    )

    runnable_exercice = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt_exercice
        | model
        | StrOutputParser()
    )
    return runnable_exercice


@cl.step(type="retrieval", name="Context via similarity_search")
def trouve_contexte(question):
    retriever = cl.user_session.get("retriever")
    search_results = retriever.vectorstore.similarity_search(question, k=20)

    # Utiliser un dictionnaire pour regrouper les chunks par source
    results_by_source = {}
    for result in search_results:
        source = result.metadata["source"]
        if source not in results_by_source:
            results_by_source[source] = []
        results_by_source[source].append(result)

    # Limiter à 5 sources différentes et récupérer plusieurs chunks par source
    relevant_sources = list(results_by_source.keys())[:5]
    # Récupérer jusqu'à 5 chunks par source
    relevant_results = [results_by_source[source][:10] for source in relevant_sources]

    # Aplatir la liste des résultats
    relevant_results = [chunk for sublist in relevant_results for chunk in sublist]

    filenames = [result.metadata["source"] for result in relevant_results]
    short_filenames = [os.path.basename(file) for file in filenames]
    print("Files used for context:", short_filenames)
    context = "\n".join(
        [
            f"-----\nSource: {os.path.basename(result.metadata['source'])}\n{result.page_content}"
            for result in relevant_results
        ]
    )

    print("-------------------\n" + context)
    return context


@cl.on_chat_start
async def factory():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))

    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(
            index_path, embeddings=embeddings_OL, allow_dangerous_deserialization=True
        )
        print("Index chargé à partir du chemin existant.")
    else:
        # Load all documents individually
        documents = load_documents_from_directory("differents_textes")

        # Initialize CharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=150
        )

        # Split each document into chunks
        chunks = []
        for doc in documents:
            splits = text_splitter.create_documents(
                text_splitter.split_text(doc["content"])
            )
            for split in splits:
                split.metadata = {"source": doc["source"], **doc.get("metadata", {})}
                chunks.append(split)

        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings_OL)

        vectorstore.save_local(index_path)
        print("Nouvel index créé et sauvegardé.")

    retriever = vectorstore.as_retriever()
    cl.user_session.set("retriever", retriever)


@cl.on_message
async def main(message):
    memory = cl.user_session.get("memory")
    question = message.content
    print("Question:" + question)

    # setup_model() et trouve_contexte() à adapter suivant ce qui est recherché
    runnable_model = setup_model()
    msg = cl.Message(content="")
    async for chunk in runnable_model.astream(
        {"question": question, "context": trouve_contexte(question)},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
