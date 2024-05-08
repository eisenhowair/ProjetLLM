import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from typing import List
import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig

# Configuration
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
index_path = "data/vectorstore/temp-index.faiss"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
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
    faiss_index.save_local(index_path)


def get_faiss_index(documents: List[str] = None) -> FAISS:
    global faiss_index
    if faiss_index is None:
        if os.path.exists(index_path):
            print("Loading existing index...")
            faiss_index = FAISS.load_local(index_path, embeddings)
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
            faiss_index.save_local(index_path)
    elif documents:
        print("Updating index with new documents...")
        update_faiss_index(faiss_index, documents)
    return faiss_index


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

    # Set up the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=faiss_index.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    cl.user_session.set("chain", qa_chain)

    msg = cl.Message(content="The bot is initialized. Ask your questions!")
    await msg.send()


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
