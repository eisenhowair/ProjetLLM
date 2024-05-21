import time
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
import chainlit as cl


llm_local = Ollama(base_url="http://localhost:11434", model="llama3")
embedding = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")


start = time.time()#

# Load blog post
loader = TextLoader("./liam.txt")
data = loader.load()
loader = TextLoader("./emma.txt")
data2 = loader.load()
loader=TextLoader('./ammamellen.txt')
data3 = loader.load()
    
docs = data + data2 + data3


end = time.time()#
timee = end - start#
print("Chargement de tous les documents=",timee)#

#PDFLoader
# import PyPDF2

# pdf = PyPDF2.PdfReader("liam.pdf")
# pdf_text = ""
# for page in pdf.pages:
#     pdf_text += page.extract_text()

# pdf = PyPDF2.PdfReader("liam.pdf")
# pdf_text2 = ""
# for page in pdf.pages:
#     pdf_text2 += page.extract_text()

# docs = pdf_text + pdf_text2





@cl.on_chat_start
async def factory():

    

    ####SIMILARITY_SEARCH
    start = time.time()#

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)


    vectorstore = Chroma(
        collection_name="full_documents", embedding_function=embedding
    )

    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    retriever.add_documents(docs, ids=None)

    end = time.time()#
    timee = end - start#
    print("embedding similarity_search=",timee)#



@cl.on_message
async def main(message):
    msg = cl.Message(content="")

    start = time.time()#

    res_similarity_search = vectorstore.similarity_search("Comment s'appelle le fils de la servante ?")

    end = time.time()#
    timee = end - start#
    print("similarity_search()=",timee)#


    start = time.time()#  

    sources = set()

    for block in res_similarity_search:
        sources.add(block.metadata["source"])

    #
    print("Sources :")
    for b in sources:
        print(b)


    ####INVOKE
        
    #load relevants documents
    docs = []
    for source in sources:
        loader = TextLoader(source)
        data = loader.load()
        docs.extend(data)

    end = time.time()#
    timee = end - start#
    print("Chargement des documents pertinents=",timee)#



    start = time.time()#    
        
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)


    vectorstore = Chroma(
        collection_name="full_documents", embedding_function=embedding
    )

    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    retriever.add_documents(docs, ids=None)

    end = time.time()#
    timee = end - start#
    print("embedding invoke=",timee)#


    start = time.time()#

    res_invoke = retriever.invoke("Comment s'appelle le fils de la servante ?")

    end = time.time()#
    timee = end - start#
    print("invoke()=",timee)#


    start = time.time()#
    

    system_instructions = """
    Vous êtes un assistant français. Votre but est de répondre aux questions à l'aide des Sources qu'on vous donnent.
    """

    prompt_template = PromptTemplate(
        template="{instructions}\n\nPrompt: {prompt}",
        input_variables=["instructions", "prompt"],
    )

    prompt = "Comment s'appelle le fils de la servante ? Sources :"
    for r in res_invoke:
        prompt += r.page_content

    full_prompt = prompt_template.format(instructions=system_instructions, prompt=prompt)
    response = llm_local.generate([full_prompt])

    end = time.time()#
    timee = end - start#
    print("Chargement modèle et génération réponse=",timee)#

    print(response.generations[0][0].text)