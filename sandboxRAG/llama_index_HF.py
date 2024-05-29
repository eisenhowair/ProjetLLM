from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import faiss
import os

# Configuration
persist_dir = "llama_index"
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
)
llm = Ollama(base_url="http://localhost:11434", model="llama3:instruct")

d = 768
faiss_index = faiss.IndexFlatL2(d)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Charger l'index si disponible
index_path = os.path.join(persist_dir, 'vector_store.index')
if os.path.exists(index_path):
    index = VectorStoreIndex.load(storage_context, persist_dir)
else:
    # Créer un nouvel index si non disponible
    reader = SimpleDirectoryReader("differents_textes/moodle")
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True)
    index.storage_context.persist(persist_dir)

# Utilisation de l'index pour une requête
query = "qui est joel heinis?"
response = index.as_query_engine(similarity_top_k=3).query(query)
print(response.response)

for node in response.source_nodes:
    print(f"{node.get_score()} 👉 {node.text}")


"""
# Commented out IPython magic to ensure Python compatibility.
# %pip install llama-index-vector-stores-faiss
# !pip install llama-index
# pip install torch transformers python-pptx Pillow
# pip install docx2txt

import faiss
from IPython.display import Markdown, display
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
import sys
import logging

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import faiss
import os

# Creating a Faiss Index

# Load documents, build the VectorStoreIndex

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

mpnet_embeddings = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)
d = 768
faiss_index = faiss.IndexFlatL2(d)


# Download Data


# load documents
documents = SimpleDirectoryReader("differents_textes/moodle/").load_data()

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# save index to disk
index.storage_context.persist()

# load index from disk
vector_store = FaissVectorStore.from_persist_dir("./storage")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage"
)
index = load_index_from_storage(storage_context=storage_context)

# Query Index

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

display(Markdown(f"<b>{response}</b>"))

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author do after his time at Y Combinator?"
)

display(Markdown(f"<b>{response}</b>"))
"""
