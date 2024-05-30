from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, load_index_from_storage
from llama_index.readers.github import GithubRepositoryReader, GithubClient
#from IPython.display import Markdown, display
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import faiss
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

REPO_NAME = os.getenv("REPO_NAME")
REPO_OWNER = os.getenv("REPO_OWNER")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BRANCH = "main"

github_client = GithubClient(github_token=GITHUB_TOKEN, verbose=True)

documents = GithubRepositoryReader(
    github_client=github_client,
    owner=REPO_OWNER,
    repo=REPO_NAME,
    use_parser=False,
    verbose=False,
    filter_directories=(
        ["sandboxRAG/differents_textes", "sandboxRAG2"],
        GithubRepositoryReader.FilterType.EXCLUDE,
    ),
    filter_file_extensions=(
        [
            ".png",
            ".txt",
            ".jpg",
            ".jpeg",
            ".py",
            ".md",
            "json",
            ".ipynb",
        ],
        GithubRepositoryReader.FilterType.INCLUDE,
    ),
).load_data(branch=BRANCH)
print(documents)
###
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
)
llm = Ollama(base_url="http://localhost:11434",
             model="llama3:instruct", request_timeout=1000.0)

d = 768
faiss_index = faiss.IndexFlatL2(d)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900

def charge_index(index_path):
    if os.path.exists(index_path):
        vector_store = FaissVectorStore.from_persist_dir(index_path)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=index_path)
        index = load_index_from_storage(storage_context=storage_context)
        return index
        storage_context = StorageContext.from_defaults(
            persist_dir=index_path
        )
        print("Index existant chargé")
        return load_index_from_storage(storage_context)
    else:
        # Créer un nouvel index si non disponible
        os.makedirs(index_path)
        print("Création d'un nouvel index")
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context_global = StorageContext.from_defaults(
            vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context_global, show_progress=True)
        index.storage_context.persist(index_path)
    return index

index = charge_index("index_github")

query_engine = index.as_query_engine()
response = query_engine.query(
    "Quelles sont les différences entre un RAG langchain et un RAG llama_index?",
    # verbose=True,
)
print(response.response)

# display(Markdown(f"{response}"))

"""
https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/ (doc officielle)
https://github.com/joaomdmoura/crewAI
https://docs.crewai.com/tools/GitHubSearchTool (impossible d'importer crewAI)
https://lightning.ai/lightning-ai/studios/chat-with-your-code-using-rag (a donné des pistes)
"""
