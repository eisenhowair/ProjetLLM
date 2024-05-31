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
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext
import faiss
import chainlit as cl

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

def create_prompt(user_query):
    return """
    You are a knowledgeable assistant with access to the content of a GitHub repository. Your task is to help users answer any questions they have about the repository, providing detailed explanations and code snippets where necessary. Follow these guidelines:

    1. Understand the User Query: Carefully read the user's question to understand what information they are seeking. Identify if the question pertains to specific files, functions, or overall project documentation within the repository.

    2. Locate Relevant Information: Search through the repository content to find relevant information that addresses the user's query. This might include reading through the README file, specific code files, comments, documentation, or any other relevant parts of the repository.

    3. Provide Detailed Explanations: Formulate a comprehensive answer that thoroughly addresses the user's question. Ensure that your explanation is clear and easy to understand, even for those who might not be familiar with the codebase.

    4. Include Code Snippets: If the question can be best answered with examples from the code, include relevant code snippets in your response. Highlight important sections of the code that are directly related to the user's query.

    5. Clarify and Elaborate: If the user’s question is broad or unclear, ask clarifying questions to better understand their needs. Provide additional context or elaboration as needed to ensure the user fully understands your response.

    6. Maintain Relevance and Accuracy: Ensure that all information and code snippets provided are accurate and directly relevant to the user's question. Avoid including unnecessary details that do not contribute to answering the question.

    Example Queries and Responses:

    1. User Query: "Can you explain the main function in the `app.js` file?"
    - Response: "Sure, here is the main function from the `app.js` file:
        ```javascript
        function main() {
            // Initialize application
            initializeApp();

            // Set up event listeners
            setupEventListeners();

            // Start the main process
            startProcess();
        }
        ```
        This function serves as the entry point of the application. It first calls `initializeApp()` to set up necessary configurations. Then, it calls `setupEventListeners()` to handle user interactions, and finally, it starts the main process with `startProcess()`. Each of these functions is defined elsewhere in the `app.js` file."

    2. User Query: "What is the purpose of the `config.json` file?"
    - Response: "The `config.json` file is used to store configuration settings for the application. It typically includes settings such as API keys, database connections, and other customizable parameters. Here is an example snippet from the `config.json` file:
        ```json
        {
            'apiKey': 'your-api-key-here',
            'database': {
                'host': 'localhost',
                'port': 5432,
                'username': 'user',
                'password': 'password'
            }
        }
        ```
        These settings are read by the application at runtime to configure the necessary services and connections."

    Instructions for the Model:

    You are to use the above guidelines to respond to user queries about the GitHub repository. Ensure each response is accurate, detailed, and includes relevant code snippets when necessary. Always strive to provide clear and thorough explanations that help the user understand the repository better.
    
    User Query:"""+user_query+"\n - Response:"

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
            ["sandboxRAG/differents_textes"],
            GithubRepositoryReader.FilterType.EXCLUDE,
        ),
        filter_file_extensions=(
            [
                ".png",
                ".jpg",
                ".jpeg",
                ".json",
            ],
            GithubRepositoryReader.FilterType.EXCLUDE,
        ),
    ).load_data(branch=BRANCH)
    print("documents bien chargés")
    for document in documents:
        file_name = document.metadata['file_name']  # Accès au nom du fichier via metadata
        print(file_name)
    #print(documents)
    return documents


embed_model = HuggingFaceEmbedding(
    model_name=embedding_model_hf_en_mpnet,  # ici changer le modèle selon embedding_models
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


def charge_index(documents,index_path = index_en_path_mpnet): # ici changer l'index selon embedding_models
    print(index_path)
    if documents is None:
        return -1
    if os.path.exists(index_path):
        vector_store = FaissVectorStore.from_persist_dir(index_path)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=index_path)
        index = load_index_from_storage(storage_context=storage_context)

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

    service_context = ServiceContext.from_defaults(embed_model=Settings.embed_model, llm=Settings.llm,callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))    
    return index, service_context


def ask_github_index(query_engine, question):
    response = query_engine.query(
        question,
        # verbose=True,
    )
    #print(response)

    return response.response

# décommenter si le fichier est utilisé tout seul
# print("ne doit pas apparaitre en utilisant chainlit")
#print(ask_github_index(charge_index(documents= fetch_repository(repo_owner="eisenhowair",
#      repo_name="ProjetLLM")), "penses-tu que les différents RAGs ont été bien implémentés?"))


"""
https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/ (doc officielle)
https://github.com/joaomdmoura/crewAI
https://docs.crewai.com/tools/GitHubSearchTool (impossible d'importer crewAI)
https://lightning.ai/lightning-ai/studios/chat-with-your-code-using-rag (a donné des pistes)
"""
