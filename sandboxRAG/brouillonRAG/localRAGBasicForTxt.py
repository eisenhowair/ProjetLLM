from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers, Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain import PromptTemplate
from PyPDF2 import PdfReader
from datetime import datetime
from typing import Optional
from io import BytesIO
import chainlit as cl
import sys
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, TextStreamer
from langchain.prompts import PromptTemplate
import transformers
import torch

from langchain.indexes import VectorstoreIndexCreator

from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider

# environment for the app
# conda activate llama2Apps
# command to run the app
# chainlit run src/apps/localLLM_withRAG-Complete.py --port 8001 -w

prompt_template = """You are an helpful AI assistant and your name is SAHAYAK. You are kind, gentle and respectful to the user. Your job is to answer the question sent by the user in concise and step by step manner. 
If you don't know the answer to a question, please don't share false information.

Context: {context}
Question: {question}

Response for Questions asked.
answer:
"""

# model used for converting text/queries to numerical embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# path to store embeddings at vectorstore
indexpath = "data/vectorstore/"

# Initialize embeddings using HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)


@cl.on_chat_start
async def factory():
    # loads the data by the user
    files = None

    ### wait for the user to upload a data file
    while files == None:
        files = await cl.AskFileMessage(
            content="""Your personal AI asistant, SAHAYAK is ready to slog!
                     To get started:
                     
1. Upload a pdf file                     
2. Ask any questions about the file!""",
            accept={"text/plain": [".txt"]},
            max_size_mb=10,
        ).send()

    # Let the user know that the system is ready
    await cl.Message(
        content=f"""Document - `"{files[0].name}"` is uploaded and being processed!"""
    ).send()

    ### Reads and convert pdf data to text
    file = files[0]
    print("voici le type de file:", type(file))
    with open(file.path, "r", encoding="utf-8") as f:
        pdf_text = f.read()
    ### Create embeddings for the uploaded documents and store in vector store
    # Initialize a text splitter for processing long texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    # Create documents by splitting the provided texts
    documents = text_splitter.create_documents([pdf_text])
    print("longueur du texte: ", len(documents), "\n", pdf_text)
    # Create a Faiss index from the embeddings
    faiss_index = FAISS.from_documents(documents, embeddings)

    # Save the Faiss index locally
    faiss_index_path = indexpath + "temp-index"
    faiss_index.save_local(faiss_index_path)
    # Load Faiss vectorstore with embeddings created and saved earlier
    load_db = FAISS.load_local(faiss_index_path, embeddings)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    model = Ollama(base_url="http://localhost:11434", model="llama3:8b")

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=load_db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    cl.user_session.set("chain", qa_chain)

    msg = cl.Message(content="The bot is getting initialized, please wait!!!")
    await msg.send()
    msg.content = "Your personal AI Assistant. Ask questions on the documents uploaded?"
    await msg.update()


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    msg = cl.Message(content="")
    async for chunk in chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        print("Chunk received:", chunk)
        await msg.stream_token(chunk["result"])
    await msg.send()
