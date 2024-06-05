from github_recup import *
import chainlit as cl
from chainlit.input_widget import TextInput
import time
import asyncio
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from langchain import hub
from llama_index.core.prompts import LangchainPromptTemplate
from langchain_core.prompts import PromptTemplate
from IPython.display import Markdown, display


"""
Temps pris pour génération de l'index:189.7270061969757
Temps pris pour la réponse à la question:182.3755898475647
instructor_large
( chainlit ne met pas à jour ses messages ou envoie le message de la réponse, alors qu'il est bien récupéré)


Temps pris pour génération de l'index:40.64496183395386
Temps pris pour la réponse à la question:125.97727012634277
mpnet-base-v2
"""

@cl.password_auth_callback
def auth_callback(username: str, password: str):

    if (username, password) == ("elias", "elias"):
        return cl.User(
            identifier="Elias", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
        
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))


@cl.on_chat_start
async def start():
    await cl.ChatSettings(
        [
            TextInput(id="owner", label="Owner Name",
                      placeholder="https://github.com/.../repository"),
            TextInput(id="name", label="Repo Name",
                      placeholder="https://github.com/owner/..."),
            TextInput(id="url", label="Url", placeholder="..."),

        ]
    ).send()
    
    await cl.Message(content="Index en cours de chargement...", author="Préparation").send()
    recup_index()
    await cl.Message(content="Index chargé!", author="Préparation").send()
    


@cl.on_message
async def main(message: cl.Message):
    question = message.content
    print("Question:" + question)
    start = time.time()
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine

    msg = cl.Message(content="", author="connexion nwaaaar")

    try:
        # Setting a timeout for the query execution
        res = await asyncio.wait_for(cl.make_async(query_engine.query)(question), timeout=600)

        # Logging sources
        for source in res.source_nodes:
            print(source)

        # Streaming response tokens
        for token in res.response_gen:
            await msg.stream_token(token)

        await msg.send()
    except asyncio.TimeoutError:
        error_msg = "Error: The query timed out."
        print(error_msg)
        await cl.Message(content=error_msg, author="connexion nwaaaar").send()
    except Exception as e:
        error_msg = f"Error while processing the query: {str(e)}"
        print(error_msg)
        await cl.Message(content=error_msg, author="connexion nwaaaar").send()

    end = time.time()
    print(f"-----\nTemps pris pour la réponse à la question: {end - start}")

    
        
@cl.step(type="run", name="Récupération de l'index")
def recup_index_old(settings):
    
    documents_repo = fetch_repository(
            repo_name=settings["name"], repo_owner=settings["owner"], repo_url=settings["url"])

    if documents_repo == -1:
        print("problème pour le dépot git")
    else:
        index,service_context = charge_index(documents=documents_repo)
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=2, service_context=service_context)
        print("index bien chargé")
        cl.user_session.set("query_engine", query_engine)


def recup_index(settings=None):

    index, service_context = charge_index(settings=settings)
    query_engine = index.as_query_engine(
        streaming=True, similarity_top_k=4, service_context=service_context)
    langchain_prompt = hub.pull("rlm/rag-prompt")
    
    display_prompt_dict(query_engine.get_prompts())
    lc_prompt_tmpl = LangchainPromptTemplate(
        template=PromptTemplate.from_template(template=create_prompt_simplifie()),
        #template=langchain_prompt,
        template_var_mappings={
            "query_str": "question", "context_str": "context"},
    )

    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": lc_prompt_tmpl}
    )
    display_prompt_dict(query_engine.get_prompts())
    print("index bien chargé")
    cl.user_session.set("query_engine", query_engine)


@cl.on_settings_update
async def setup_agent(settings):
    start = time.time()
    recup_index(settings=settings)
    end = time.time()
    print(f"-----\nTemps pris pour génération de l'index:{end-start}")
