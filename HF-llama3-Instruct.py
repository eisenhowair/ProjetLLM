from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, TextStreamer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import transformers
import torch
import ollama
import os
import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from transformers import LlamaForCausalLM
from ollama import Llama


chemin_cache_modele = "/home/UHA/e2303253/U/modeleLLM"
os.environ["TRANSFORMERS_CACHE"] = chemin_cache_modele

template = """
You are a helpful AI assistant. Provide the answer for the following question:

Question: {question}
Answer:from ollama import Llama
"""

# You need to be approved by HF & Meta https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main


@cl.cache
def load_llama():
    
    model_path = "/usr/share/ollama/.ollama/models"
    llama_model = ollama.load_model_from_manifest(model_path + "/manifests/registry.ollama.ai")
    tokenizer = ollama.get_tokenizer(llama_model)

    # Create a custom HuggingFacePipeline instance
    pipeline = transformers.pipeline(
        "text-generation",
        model=llama_model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(
        pipeline=pipeline,
        model_kwargs={"temperature": 0},
    )
    return llm


llm = load_llama()

add_llm_provider(
    LangchainGenericProvider(
        id=llm._llm_type, name="Llama3-Instruct-chat", llm=llm, is_chat=False
    )
)


@cl.on_chat_start
async def main():

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    # print(llm_chain.run("A quel âge est décédé Einstein?"))
    cl.user_session.set("llm_chain", llm_chain)  # on assigne le modèle à la session

    return llm_chain


@cl.on_message
async def run(message: cl.Message):
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer"]
    )

    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get(
        "llm_chain"
    )  # type: LLMChain 
    #on récupère le modèle ici
    res = await llm_chain.acall(
        message.content, callbacks=[cb]
    )  # puis on applique le modèle à la question

    if not cb.answer_reached:
        await cl.Message(content=res["text"]).send()
