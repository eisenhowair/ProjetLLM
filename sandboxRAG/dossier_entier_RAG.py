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

model = Ollama(base_url="http://localhost:11434", model="llama3:8b")


# Global index variable
faiss_index = None


# Function to read PDF and return text
def read_text_from_file(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    elif file_path.lower().endswith(".json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.dumps(json.load(f))
    else:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")


# Function to load documents individually
def load_documents_from_directory(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith((".txt", ".pdf",".json")):
                print("Traitement de ", filename)
                file_path = os.path.join(root, filename)
                try:
                    content = read_text_from_file(file_path)
                    documents.append({"content": content, "source": file_path})
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
            """Tu parles uniquement français. Ton rôle est de répondre à la question de l'utilisateur en te basant **uniquement** sur le contexte fourni. 
            Si tu ne trouves pas la réponse dans le contexte, dis-le clairement au lieu de deviner. 
            Contexte: {context}"""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ]
)

    runnable_exercice = (
        RunnablePassthrough.assign(
            history=RunnableLambda(
                memory.load_memory_variables) | itemgetter("history")
        ) 
        | prompt_exercice
        | model
        | StrOutputParser()
    )
    return runnable_exercice


@cl.step(type="retrieval", name="Context via similarity_search")
def trouve_contexte(question):

    retriever = cl.user_session.get("retriever")
    search_results = retriever.vectorstore.similarity_search(question, k=5)
    filenames = [result.metadata["source"] for result in search_results]
    print("Files used for context:", filenames)
    context = "\n".join([result.page_content for result in search_results])

    return context


@cl.on_chat_start
async def factory():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))

    # Load all documents individually
    documents = load_documents_from_directory("differents_textes")

    # Initialize CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=40)

    # Split each document into chunks
    chunks = []
    for doc in documents:
        splits = text_splitter.create_documents(text_splitter.split_text(doc["content"]))
        for split in splits:
            split.metadata = {"source": doc["source"]}
            chunks.append(split)

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(
            base_url="http://localhost:11434",
            model="nomic-embed-text",
            show_progress="true",
            temperature=0,
        ),
    )

    vectorstore.save_local(index_path)

    retriever = vectorstore.as_retriever()
    cl.user_session.set("retriever",retriever)
    # Set up the QA chain
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    cl.user_session.set("chain", qa_chain)

    question_manuelle = input("Question:")
    """


    #manual_test()
    
    
def manual_test():

    retriever = cl.user_session.get("retriever")
    question_manuelle="Comment s'appelle le boss final de Elias adventure"
    search_results = retriever.vectorstore.similarity_search(question_manuelle, k=5)
    filenames = [result.metadata["source"] for result in search_results]
    print("Files used for context:", filenames)
    context = "\n".join([result.page_content for result in search_results])
    # context = search_results[0].page_content
    print("Question:", question_manuelle)
    rag_data = {
        "context": context,
        "question": question_manuelle,
    }
    
    # use the context to answer the question
    prompt_exercice = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu parles uniquement français. Ton rôle est de répondre à la question de l'utilisateur en te basant uniquement sur le contexte,\
                Contexte: {context}"
            ),
            ("human", "{question}")
        ]
    )
    rag_chain = RunnablePassthrough() | prompt_exercice | model | StrOutputParser()
    print("Answer:", rag_chain.invoke(rag_data))

@cl.on_message
async def main(message):
    #chain = cl.user_session.get("chain")
    memory= cl.user_session.get("memory")


    # setup_model() et trouve_contexte() à adapter suivant ce qui est recherché
    runnable_model = setup_model()
    msg = cl.Message(content="")
    async for chunk in runnable_model.astream(
        {"question": message.content,"context": trouve_contexte(message.content)},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    #memory.chat_memory.add_user_message(message.content)
    #memory.chat_memory.add_ai_message(msg.content)
    await msg.send()
