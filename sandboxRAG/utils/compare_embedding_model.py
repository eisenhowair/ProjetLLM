import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

from embedding_models import *


def read_text_from_file(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            metadata = reader.metadata
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text, metadata
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), {}
    elif file_path.lower().endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.dumps(json.load(f)), {}
    else:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")


# Function to load documents individually
def load_documents_from_directory(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith((".txt", ".pdf", ".json")):
                print("Traitement de ", filename)
                file_path = os.path.join(root, filename)
                try:
                    text, metadata = read_text_from_file(file_path)
                    documents.append(
                        {"content": text, "source": file_path, "metadata": metadata}
                    )
                except ValueError as e:
                    print(f"Error processing {file_path}: {e}")
    return documents


# Initialize embeddings
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
ollama_embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text",
    show_progress="true",
    temperature=0,
)
distilbert_embeddings = HuggingFaceEmbeddings(
    model_name="distilbert-base-nli-stsb-mean-tokens"
)
instructor_embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
camembert_embeddings = HuggingFaceEmbeddings(
    model_name="dangvantuan/sentence-camembert-base"
)
mpnet_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)


def charge_vectorstore(embedding, index_path):
    # pour se mettre au niveau du répertoire au dessus contenant les vectorstores
    index_path = "../" + str(index_path)
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(
            index_path,
            embeddings=embedding,
            # allow_dangerous_deserialization=True
        )
        print(f"Index chargé à partir du chemin existant: {index_path}")
    else:
        documents = load_documents_from_directory("../differents_textes")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=150
        )
        chunks = []
        for doc in documents:
            splits = text_splitter.create_documents(
                text_splitter.split_text(doc["content"])
            )
            for split in splits:
                split.metadata = {"source": doc["source"], **doc.get("metadata", {})}
                chunks.append(split)
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding)
        vectorstore.save_local(index_path)
        print(f"Nouvel index créé et sauvegardé: {index_path}")
    return vectorstore


# Function to test retrieval and track usage
def test_retrieval(question, path_vectorstore, embedding, model_name):
    vectorstore = charge_vectorstore(embedding, path_vectorstore)
    retriever = vectorstore.as_retriever()
    search_results = retriever.vectorstore.similarity_search(question, k=15)

    # Count usage of each source
    source_count = {}
    for result in search_results:
        source = result.metadata["source"]
        if source not in source_count:
            source_count[source] = 0
        source_count[source] += 1

    context = "\n--\n".join(
        [
            f"Source: {result.metadata['source']}\n{result.page_content}"
            for result in search_results
        ]
    )

    # print(        f"---------\nContext for question '{question}' using {model_name}:\n{context}"    )
    print(f"File usage count for {model_name}:")
    for source, count in source_count.items():
        print(f"{os.path.basename(source)}: {count} times")

    return source_count


# Function to plot usage
def plot_usage(data, question):
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    models = list(data.keys())
    all_sources = set()
    for model_data in data.values():
        all_sources.update(model_data.keys())

    all_sources = sorted(all_sources)
    usage_matrix = []
    for model in models:
        model_data = data[model]
        usage_counts = [model_data.get(source, 0) for source in all_sources]
        usage_matrix.append(usage_counts)

    usage_matrix = list(zip(*usage_matrix))  # Transpose the matrix

    fig, ax = plt.subplots(figsize=(14, 8))
    x = range(len(models))
    for i, source in enumerate(all_sources):
        ax.bar(x, [usage_matrix[i][j] for j in x], label=os.path.basename(source))

    ax.set_xlabel("Models")
    ax.set_ylabel("Number of times source used")
    ax.set_title(f"Source Usage Comparison for Question: {question}")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(title="Sources")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "../screenshot_compare_embedding", f"{question.replace(' ', '_')}.png"
        )
    )
    plt.show()


# Example question
# question = "Parle moi de ce que Aleister Crowley est allé faire en Egypte"
question = input("Posez une question:")
# Collect data for all models
usage_data = {}

print("Using HuggingFace Embeddings (all-MiniLM-L6-v2):")
usage_data["all-MiniLM-L6-v2"] = test_retrieval(
    question,
    "vectorstores/HF_vectorstore",
    hf_embeddings,
    "HuggingFace Embeddings (all-MiniLM-L6-v2)",
)

print("\nUsing Ollama Embeddings:")
usage_data["nomic-embed-text"] = test_retrieval(
    question, index_en_path_OL, ollama_embeddings, "Ollama Embeddings"
)


print("\nUsing DistilBERT Embeddings (distilbert-base-nli-stsb-mean-tokens):")
usage_data["distilbert-base-nli-stsb-mean-tokens"] = test_retrieval(
    question,
    "vectorstores/DistilBERT_vectorstore",
    distilbert_embeddings,
    "DistilBERT Embeddings (distilbert-base-nli-stsb-mean-tokens)",
)

print("\nUsing Instructor Base Embeddings (hkunlp/instructor-base):")
usage_data["hkunlp/instructor-base"] = test_retrieval(
    question,
    index_en_path_instructor_base,
    HuggingFaceEmbeddings(model_name=embedding_model_hf_en_instructor_base),
    "Instructor Embeddings (hkunlp/instructor-base)",
)
"""
print("\nUsing Instructor xl Embeddings (hkunlp/instructor-xl):")
usage_data["hkunlp/instructor-xl"] = test_retrieval(
    question,
    index_en_path_instructor_xl,
    HuggingFaceEmbeddings(model_name=embedding_model_hf_en_instructor_xl),
    "Instructor Embeddings (hkunlp/instructor-xl)",
)
"""
print("\nUsing Instructor Embeddings (hkunlp/instructor-large):")
usage_data["hkunlp/instructor-large"] = test_retrieval(
    question,
    index_en_path_instructor_large,
    instructor_embeddings,
    "Instructor Embeddings (hkunlp/instructor-large)",
)

print("\nUsing MPNet Embeddings (sentence-transformers/all-mpnet-base-v2):")
usage_data["sentence-transformers/all-mpnet-base-v2"] = test_retrieval(
    question,
    index_en_path_mpnet,
    mpnet_embeddings,
    "MPNet Embeddings (sentence-transformers/all-mpnet-base-v2)",
)

print("\nUsing Camembert Embeddings (dangvantuan/sentence-camembert-base):")
usage_data["dangvantuan/sentence-camembert-base"] = test_retrieval(
    question,
    index_fr_path_camembert,
    mpnet_embeddings,
    "Camembert Embeddings (dangvantuan/sentence-camembert-base)",
)

# Plot usage data
plot_usage(usage_data, question)
