import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sandboxRAG.utils.embedding_models import *

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, load_index_from_storage
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import faiss

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

REPO_NAME = os.getenv("REPO_NAME")
REPO_OWNER = os.getenv("REPO_OWNER")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BRANCH = "main"


def fetch_repository(repo_owner=None, repo_name=None, repo_url=None):
    if repo_url:
        parts = repo_url.split('/')
        repo_owner = parts[-2]
        repo_name = parts[-1]
    elif repo_name is None or repo_owner is None:
        print("dépôt invalide")
        return -1  # car l'url n'est pas là non plus

    github_client = GithubClient(github_token=GITHUB_TOKEN, verbose=True)

    documents = GithubRepositoryReader(
        github_client=github_client,
        owner=repo_owner,
        repo=repo_name,
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
    print("documents bien chargés")

    return documents


embed_model = HuggingFaceEmbedding(
    model_name=embedding_model_hf_en_instructor_large,  # ici changer le modèle selon embedding_models
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


def charge_index(documents,index_path = index_en_path_instructor_large): # ici changer l'index selon embedding_models
    print(index_path)
    if documents is None:
        return -1
    if os.path.exists(index_path):
        vector_store = FaissVectorStore.from_persist_dir(index_path)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=index_path)
        index = load_index_from_storage(storage_context=storage_context)
        return index

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


def ask_github_index(index, question):
    query_engine = index.as_query_engine()
    response = query_engine.query(
        question,
        # verbose=True,
    )
    print(response)

    return response.response

# décommenter si le fichier est utilisé tout seul
# print("ne doit pas apparaitre en utilisant chainlit")
# print(ask_github_index(charge_index("index_github", fetch_repository(repo_owner="eisenhowair",
#      repo_name="ProjetLLM")), "penses-tu que les différents RAGs ont été bien implémentés?"))


"""
https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/ (doc officielle)
https://github.com/joaomdmoura/crewAI
https://docs.crewai.com/tools/GitHubSearchTool (impossible d'importer crewAI)
https://lightning.ai/lightning-ai/studios/chat-with-your-code-using-rag (a donné des pistes)
"""
