from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

chemin_cache_modele = "/home/UHA/e2303253/U/modeleLLM"
os.environ["TRANSFORMERS_CACHE"] = chemin_cache_modele

template = """
You are a helpful AI assistant. Provide the answer for the following question:

Question: {question}
Answer:
"""

@cl.cache
def load_llama():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)

    return model, tokenizer

model, tokenizer = load_llama()

def generate_text(question, model, tokenizer):
    inputs = tokenizer.encode("Question: " + question + "\nAnswer:", return_tensors="pt")
    output_sequences = model.generate(
        input_ids=inputs,
        max_length=100,
        temperature=0.7,
        top_k=10,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

llm = HuggingFacePipeline(
    model=model,
    tokenizer=tokenizer,
    generate_func=generate_text,  # Suppose HuggingFacePipeline can accept a custom generate function
)

add_llm_provider(
    LangchainGenericProvider(
        id="meta-llama-3-8b-instruct",
        name="Llama3-Instruct-chat",
        llm=llm,
        is_chat=False
    )
)

@cl.on_chat_start
async def main():

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    cl.user_session.set("llm_chain", llm_chain)  # on assigne le modèle à la session

    return llm_chain

@cl.on_message
async def run(message: cl.Message):
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer"]
    )

    llm_chain = cl.user_session.get("llm_chain")
    response = generate_text(message.content, model, tokenizer)  # Utilisez ici la fonction generate_text
    if not cb.answer_reached:
        await cl.Message(content=response).send()