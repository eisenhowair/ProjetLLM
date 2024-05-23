import ollama
import time
import os
import json
import numpy as np
from numpy.linalg import norm
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import BaseEmbedding

class OllamaEmbeddings(BaseEmbedding):
    def __init__(self, model="llama-3"):
        self.model = model

    def embed_documents(self, texts):
        embeddings = [ollama.embeddings(model=self.model, prompt=text)["embedding"] for text in texts]
        return np.array(embeddings)

    def embed_query(self, text):
        return ollama.embeddings(model=self.model, prompt=text)["embedding"]

# open a file and return paragraphs
def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
    return paragraphs

def main():
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
based on snippets of text provided in context. Answer only using the context provided,
being as concise as possible. If you're unsure, just say that you don't know.
Context:
"""

    # open file
    filename = "data/peter-pan.txt"
    paragraphs = parse_file(filename)
    paragraphs = " ".join(paragraphs)

    # load documents and create vector store
    loader = TextLoader(filename)
    documents = loader.load()
    embeddings = ollama.embeddings(model="nomic-embed-text", prompt=paragraphs)
    # vectorstore = Chroma.from_documents(documents, embeddings)
    vectorstore = Chroma.from_documents(documents, embeddings)

    prompt = input("what do you want to know? -> ")

    # find most similar documents
    similar_docs = vectorstore.similarity_search(prompt, k=5)
    context = '\n'.join([doc.page_content for doc in similar_docs])

    # get response from Ollama model
    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + context,
            },
            {"role": "user", "content": prompt},
        ],
    )

    print("\n\n")
    print(response["message"]["content"])

if __name__ == "__main__":
    main()