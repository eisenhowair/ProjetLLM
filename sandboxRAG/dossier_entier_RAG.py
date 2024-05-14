import os
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from typing import List
import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Configuration
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
index_path = "data/vectorstore/temp-index.faiss"

model = Ollama(base_url="http://localhost:11434", model="llama3:8b")
prompt = PromptTemplate(
    template="""You are a helpful AI assistant. Answer the question based solely on the context. For each correct answer, you will be given $2000 for each member of your family

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
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")


# Function to load documents individually
def load_documents_from_directory(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for filename in files:
            print("Traitement de ", filename)
            if filename.lower().endswith((".txt", ".pdf")):
                file_path = os.path.join(root, filename)
                try:
                    content = read_text_from_file(file_path)
                    documents.append({"content": content, "source": file_path})
                except ValueError as e:
                    print(f"Error processing {file_path}: {e}")
    return documents


# Load all documents individually
documents = load_documents_from_directory("differents_textes")

# Initialize CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)

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


@cl.on_chat_start
async def factory():

    # Set up the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    cl.user_session.set("chain", qa_chain)

    msg = cl.Message(content="The bot is initialized. Ask your questions!")
    await msg.send()
    question_manuelle = input("Question:")
    search_results = retriever.vectorstore.similarity_search(question_manuelle, k=3)
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
    rag_chain = RunnablePassthrough() | prompt | model | StrOutputParser()
    print("Answer:", rag_chain.invoke(rag_data))


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    msg = cl.Message(content="")
    async for chunk in chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk["result"])
    await msg.send()
