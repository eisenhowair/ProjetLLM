from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, TextStreamer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import transformers
import torch
import chainlit as cl
from chainlit.playground.config import add_llm_provider
from chainlit.playground.providers.langchain import LangchainGenericProvider

# ce programme nécessite un compte HuggingFace disposant d'une autorisation
# pour utiliser le modèle suivant depuis HF: Meta-Llama-3-8B-Instruct
# après avoir demandé et obtenu l'autorisation, aller générer un token depuis votre profil HF
# et l'écrire dans le terminal lorsqu'il sera demandé après y avoir écrit "huggingface-cli login"

template = """
You are a helpful AI assistant. Provide the answer for the following question:

Question: {question}
Answer:
"""


@cl.cache
def load_llama():

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    # custom HuggingFacePipeline instance
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
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
        id=llm._llm_type, name="Llama3-Instruct-chat", llm=llm, is_chat=True
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

    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    # on récupère le modèle ici
    res = await llm_chain.acall(
        message.content, callbacks=[cb]
    )  # puis on applique le modèle à la question

    if not cb.answer_reached:
        await cl.Message(content=res["text"]).send()
