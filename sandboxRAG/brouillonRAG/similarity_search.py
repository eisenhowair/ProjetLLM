import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from typing import List
import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
import os
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding

# Configuration
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
index_path = "data/vectorstore/temp-index.faiss"

# Ensure the directory exists
os.makedirs(os.path.dirname(index_path), exist_ok=True)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)
model = Ollama(base_url="http://localhost:11434", model="llama3:8b")
prompt = PromptTemplate(
    template="""You are a helpful AI assistant named SAHAYAK. Answer the question based on the context.

Context: {context}
Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

# Global index variable
faiss_index = None


# Function to read PDF and return text
def read_text_from_file(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")


def update_faiss_index(faiss_index: FAISS, documents: List[str]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    new_docs = text_splitter.create_documents(documents)
    faiss_index.add_documents(new_docs)
    faiss_index.save_local(index_path, index_name="tryoutIndex")


def get_faiss_index(documents: List[str] = None) -> FAISS:
    global faiss_index
    if faiss_index is None:
        if os.path.exists(os.path.join(index_path, "tryoutIndex.faiss")):
            print("Loading existing index...")
            faiss_index = FAISS.load_local(
                index_path, embeddings, index_name="tryoutIndex"
            )
            if documents:
                update_faiss_index(faiss_index, documents)
        else:
            print("Creating new index...")
            if documents is None:
                raise ValueError("No documents provided to create an index.")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=10
            )
            docs = text_splitter.create_documents(documents)
            faiss_index = FAISS.from_documents(docs, embeddings)
            faiss_index.save_local(index_path, index_name="tryoutIndex")
    elif documents:
        print("Updating index with new documents...")
        update_faiss_index(faiss_index, documents)
    return faiss_index


def search_and_answer(question: str, retriever: FAISS, prompt: PromptTemplate) -> str:
    # Effectuer la recherche
    search_docs = retriever.similarity_search(question)

    # Préparer le contexte pour le prompt
    context = "\n\n".join([doc.page_content for doc in search_docs])

    # Générer la réponse
    formatted_prompt = prompt.format(context=context, question=question)
    answer = model(formatted_prompt)
    return answer


@cl.on_chat_start
async def factory():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="""Your personal AI assistant, SAHAYAK is ready to help!
                        To get started:
                        
1. Upload a PDF file                     
2. Ask any questions about the file!""",
            accept={"application/pdf": [".pdf"], "text/plain": [".txt"]},
            max_size_mb=10,
        ).send()

    await cl.Message(
        content=f"""Document - `"{files[0].name}"` is uploaded and being processed!"""
    ).send()

    # Read and process PDF file
    file_text = read_text_from_file(files[0].path)
    print(
        f"Processed text from {files[0].name}: {file_text[:60]}..."
    )  # Print a snippet for vérification

    # Load or update the FAISS index
    faiss_index = get_faiss_index([file_text])

    question = input("Enter your query:")
    rag_data = {"context": faiss_index.similarity_search, "question": question}
    rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
    print("Answer:", rag_chain.invoke(rag_data))
    """
    searchDocs = faiss_index.similarity_search(question)
    for doc in searchDocs:
        print("similarity_search:", doc.page_content)
    """
    # Set la chaîne de recherche dans la session utilisateur
    cl.user_session.set("retriever", faiss_index)

    msg = cl.Message(content="The bot is initialized. Ask your questions!")
    await msg.send()


@cl.on_message
async def main(message):
    retriever = cl.user_session.get("retriever")
    if retriever is None:
        await cl.Message(
            content="Error: No retriever found. Please restart the chat."
        ).send()
        return

    question = (
        message.content
    )  # Récupérer la question de l'utilisateur depuis le message Chainlit

    search_docs = retriever.similarity_search(
        question
    )  # Effectuer la recherche de similarité avec la question

    context = "\n\n".join(
        [doc.page_content for doc in search_docs]
    )  # Préparer le contexte pour le prompt

    formatted_prompt = prompt.format(
        context=context, question=question
    )  # Formater le prompt avec le contexte et la question

    answer = model(formatted_prompt)  # Générer la réponse complète

    await cl.Message(content=answer).send()  # Envoyer la réponse à l'utilisateur
