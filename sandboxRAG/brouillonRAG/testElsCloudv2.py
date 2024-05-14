from elasticsearch import Elasticsearch, helpers
import configparser
import json
from llama_index.core import Document, Settings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.elasticsearch import ElasticsearchEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

config = configparser.ConfigParser()
config.read("example.ini")

es = Elasticsearch(
    cloud_id=config["ELASTIC"]["cloud_id"],
    basic_auth=(config["ELASTIC"]["user"], config["ELASTIC"]["password"]),
)

print(es.info())
# aller dans Deployment -> Security -> security changes in kibana -> create an user with the appropriates roles

with open("conversations.json", "r") as f:
    conversations = json.load(f)

"""
embeddings = ElasticsearchEmbedding.from_credentials(
    model_id="llama3:8b",
    es_url="localhost:9200",
    es_username=config["ELASTIC"]["user"],
    es_password=config["ELASTIC"]["password"],
)
"""


# Create an OLLAMA embedding model
ollama_embedding = OllamaEmbedding("llama3:8b")
# set global settings
Settings.embed_model = ollama_embedding
Settings.chunk_size = 512
# Create an Elasticsearch store
es_store = ElasticsearchStore(
    index_name="conversations",
    es_client=es,
)

storage_context = StorageContext.from_defaults(vector_store=es_store)


index = VectorStoreIndex.from_vector_store(
    vector_store=es_store,
    storage_context=storage_context,
)

query_engine = index.as_query_engine()


response = query_engine.query("hello world")
"""
# Iterate over the conversations and create Document objects
documents = []
for conversation in conversations:
    doc = Document(
        text=conversation["conversation"],
        metadata={"conversation_id": conversation["conversation_id"]},
    )
    documents.append(doc)
"""
