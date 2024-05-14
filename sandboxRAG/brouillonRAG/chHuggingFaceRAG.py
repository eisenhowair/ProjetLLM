from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from datetime import datetime
from typing import Optional
from io import BytesIO
import chainlit as cl
import sys

# environment for the app
# conda activate llama2Apps
# command to run the app
# chainlit run src/apps/localLLM_withRAG-Complete.py --port 8001 -w

# Prompt Template
prompt_template = """You are an helpful AI assistant and your name is SAHAYAK. You are kind, gentle and respectful to the user. Your job is to answer the question sent by the user in concise and step by step manner. 
If you don't know the answer to a question, please don't share false information.
            
Context: {context}
Question: {question}

Response for Questions asked.
answer:
"""
# Model path and embedding model
modelpath = "../models/llama-2-7b-chat.Q2_K.gguf"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize embeddings using HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Model parameters
# path to store embeddings at vectorstore
indexpath = "data/vectorstore/"
# number of neural network layers to be transferred to be GPU for computation
n_gpu_layers = 10
n_batch = 256

config = {
    "max_new_tokens": 512,
    "context_length": 4096,
    "gpu_layers": n_gpu_layers,
    "batch_size": n_batch,
    "temperature": 0.1,
}


@cl.on_chat_start
# Actions to be taken once the RAG app starts
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
            accept={"application/pdf": [".pdf"]},
            max_size_mb=10,
        ).send()

    # Let the user know that the system is ready
    await cl.Message(
        content=f"""Document - `"{files[0].name}"` is uploaded and being processed!"""
    ).send()

    ### Reads and convert pdf data to text
    file = files[0]
    # Convert the content of the PDF file to a BytesIO stream
    text_stream = BytesIO(file.content)
    # Create a PdfReader object from the stream to extract text
    pdf = PdfReader(text_stream)
    pdf_text = ""
    # Iterate through each page in the PDF and extract text
    for page in pdf.pages:
        pdf_text += page.extract_text()  # Concatenate the text from each page

    ### Create embeddings for the uploaded documents and store in vector store
    # Initialize a text splitter for processing long texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    # Create documents by splitting the provided texts
    documents = text_splitter.create_documents([pdf_text])
    # Create a Faiss index from the embeddings
    faiss_index = FAISS.from_documents(documents, embeddings)

    # Save the Faiss index locally
    faiss_index_path = indexpath + "temp-index"
    faiss_index.save_local(faiss_index_path)
    # Load Faiss vectorstore with embeddings created and saved earlier
    db = FAISS.load_local(faiss_index_path, embeddings)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    """
    # Create a retrievalQA chain using Llama2
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Replace with the actual chain type
        retriever=db.as_retriever(
            search_kwargs={"k": 1}
        ),  # Assuming vectorstore is used as a retriever
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    """
    model = Ollama(base_url="http://localhost:11434", model="llama3:8b")
    runnable = prompt | model | StrOutputParser()
    msg = cl.Message(content="The bot is getting initialized, please wait!!!")
    await msg.send()
    msg.content = "Your personal AI Assistant, SAHAYAK is ready. Ask questions on the documents uploaded?"
    await msg.update()
    # cl.user_session.set("chain", chain)
    cl.user_session.set("runnable", runnable)


# Actions to be taken once user send the query/message
@cl.on_message
async def main(message):
    start_time = datetime.now()
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
