from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, TextStreamer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
import transformers
import torch

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.vectorstores import ElasticVectorSearch

import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider


template = """
You are a helpful AI assistant. Provide the answer for the following question based on the given context:

Context: {context}

Question: {question}
Answer:
"""


@cl.cache
def load_llama():
    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    llm = HuggingFacePipeline(
        pipeline=pipeline,
        model_kwargs={"temperature": 0},
    )
    return llm


llm = load_llama()

add_llm_provider(
    LangchainGenericProvider(
        id=llm._llm_type, name="Llama2-chat", llm=llm, is_chat=False
    )
)


@cl.cache
def load_index():
    loader = TextLoader("donnees_uni_test.txt")
    embeddings = HuggingFaceEmbeddings()
    index = VectorstoreIndexCreator(
        vectorstore_cls=ElasticVectorSearch,
        embedding=embeddings,
        vectorstore_kwargs={
            "elasticsearch_url": "https://5d498084b1374d03923703344a873fdb.us-central1.gcp.cloud.es.io",
            "elasticsearch_api_key": "l6n4LTZYTSm7BRv-W4VXxw",
        },
    ).from_loaders([loader])
    return index


@cl.on_chat_start
async def main():
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    index = load_index()
    retriever = index.vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    cl.user_session.set("qa_chain", qa_chain)
    return qa_chain


@cl.on_message
async def run(message: cl.Message):
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer"]
    )

    qa_chain = cl.user_session.get("qa_chain")
    res = await qa_chain.acall({"query": message.content}, callbacks=[cb])

    if not cb.answer_reached:
        await cl.Message(content=res["result"]).send()
