from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.output_parsers import StrOutputParser

from langchain.docstore.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable.config import RunnableConfig
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory

import PyPDF2
from operator import itemgetter
import unicodedata
import re

import chainlit as cl


llm_local = Ollama(base_url="http://localhost:11434", model="llama3:instruct")

embedding_model="hkunlp/instructor-large"
embedding = HuggingFaceEmbeddings(model_name=embedding_model)



def clean_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def load_pdf_as_document(file_path, name=None):
    pdf = PyPDF2.PdfReader(file_path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    source = name if name else file_path
    pdf_text = clean_text(pdf_text)
    return Document(page_content=pdf_text, metadata={"source": source})



@cl.on_chat_start
async def factory():

    files = None

    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload pdf files (max 5)", accept=["pdf"], max_files=5
        ).send()

    docs = []
    for file in files:
        doc_pdf = load_pdf_as_document(file.path, file.name)
        docs.append(doc_pdf)
    
    msg = cl.Message(content="")
    await msg.send()

    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)

    vectorstore = Chroma(collection_name="full_documents", embedding_function=embedding)

    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents(docs, ids=None)

    cl.user_session.set("retriever", retriever)
    cl.user_session.set("vectorstore", vectorstore)

    msg.content = "Entrez votre question !"
    await msg.update()


@cl.step(type="retrieval", name="similarity search avec retriever")
def sim_search(question):

    vectorstore = cl.user_session.get("vectorstore")
    retriever = cl.user_session.get("retriever")
    
    res_similarity_search = vectorstore.similarity_search(question)
    res_invoke = retriever.invoke(question)

    
    sources = set()

    for block in res_similarity_search:
        sources.add(block.metadata["source"])

    for block in res_invoke:
        sources.add(block.metadata["source"])

    print("Sources :")
    for b in sources:
        print(b)

    return res_similarity_search + res_invoke


@cl.step(type="run", name="Mise en place du Runnable")
def setup_model():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """{sys_instruction}\n\nPrompt: {prompt}""",
            ),
            MessagesPlaceholder(variable_name="history"),
        ]
    )

    runnable_exercice = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | llm_local
        | StrOutputParser()
    )
    return runnable_exercice


@cl.on_message
async def main(message):

    memory = cl.user_session.get("memory")

    runnable_model = setup_model()
    res_invoke = sim_search(message.content)
    prompt = message.content + " \nSources :"
    for r in res_invoke:
        prompt += r.page_content

    system_instructions = """
    Vous êtes un assistant français. 
    Vous devez répondre uniquement à l'aide des [[sources]] qu'on vous donne.
    """

    msg = cl.Message(content="")
    async for chunk in runnable_model.astream(
        {"sys_instruction": system_instructions, "prompt": prompt},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()