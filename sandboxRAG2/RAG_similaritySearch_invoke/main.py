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
from langchain.docstore.document import Document

llm_local = Ollama(base_url="http://localhost:11434", model="llama3:8b")
embedding = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")


start = time.time()#

# Texte Loader
# loader = TextLoader("./liam.txt")
# data = loader.load()
# loader = TextLoader("./emma.txt")
# data2 = loader.load()
# loader=TextLoader('./ammamellen.txt')
# data3 = loader.load()
    
# docs = data + data2 + data3


#PDFLoader
import PyPDF2

def load_pdf_as_document(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    return Document(page_content=pdf_text, metadata={"source": file_path})

pdf_doc1 = load_pdf_as_document("Dumas_Les_trois_mousquetaires_1.pdf")
# pdf_doc2 = load_pdf_as_document("emma.pdf")
# pdf_doc3 = load_pdf_as_document("ammamellen.pdf")

docs = [pdf_doc1]


#JSON Loader
# import json
# def load_json_as_document(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         json_data = json.load(f)
#         json_text = json.dumps(json_data, ensure_ascii=False, indent=2)
#     return Document(page_content=json_text, metadata={"source": file_path})

# jsonDoc = load_json_as_document("horaires_uha.json")
# jsonDoc2 = load_json_as_document("horaires_magasins_mulhouse.json")
# docs = [jsonDoc, jsonDoc2]


end = time.time()#
timee = end - start#
print("Chargement de tous les documents=",timee)#




###SIMILARITY_SEARCH
start = time.time()#

child_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)


vectorstore = Chroma(
    collection_name="full_documents", embedding_function=embedding
)

###################################################
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




start = time.time()#

res_similarity_search = vectorstore.similarity_search("Qui est Grimaud ?")

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

#txt
# for source in sources:
#     loader = TextLoader(source)
#     data = loader.load()
#     docs.extend(data)

#pdf
for source in sources:
    data = load_pdf_as_document(source)
    docs.append(data)

#json
# for source in sources:
#     data = load_json_as_document(source)
#     docs.append(data)


end = time.time()#
timee = end - start#
print("Chargement des documents pertinents=",timee)#



start = time.time()#    
    
child_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)


vectorstore = Chroma(
    collection_name="relevants_documents", embedding_function=embedding
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

res_invoke = retriever.invoke("Qui est Grimaud ?")

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

prompt = "Qui est Grimaud ? Sources :"
for r in res_invoke:
    prompt += r.page_content

print(prompt)

full_prompt = prompt_template.format(instructions=system_instructions, prompt=prompt)
response = llm_local.generate([full_prompt])

end = time.time()#
timee = end - start#
print("Chargement modèle et génération réponse=",timee)#

print(response.generations[0][0].text)