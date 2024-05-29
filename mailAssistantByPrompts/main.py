from langchain_community.llms import Ollama
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory

import PyPDF2

from operator import itemgetter

import chainlit as cl
from chainlit.input_widget import TextInput, Select


llm_local = Ollama(base_url="http://localhost:11434", model="llama3:instruct")
    
@cl.on_chat_start
async def start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))


    await cl.ChatSettings(
        [
            TextInput(id="indications", label="Indications", initial=""),
            Select(
                id="ton",
                label="Ton de la réponse",
                values=["Formel", "Enjoué",
                        "Familier", "Désolé"],
                initial_index=0,
            ),
        ]
    ).send()

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)


@cl.step(type="run", name="Mise en place du Runnable")
def setup_model():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory



    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """{sys_instruction}\n\nPrompt: {prompt}""",
            ),
            MessagesPlaceholder(variable_name="history"),
        ]
    )

    runnable_exercice = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | llm_local
        | StrOutputParser()
    )
    return runnable_exercice



@cl.on_message
async def main(msg: cl.Message):

    settings = cl.user_session.get("settings")

    system_instructions = """
    Vous êtes un assistant français. 
    """

    if settings :
        if settings["ton"]:
            ton = f"""Vous devez répondre avec un ton {settings["ton"]}."""

        if settings["indications"]:
            user_indications = f"""Vous devez répondre aux mails avec les [[indications]].\nIndications : {settings["indications"]}."""

    else:
        ton = "Vous devez répondre avec un ton formel."
        user_indications = "Vous devez répondre aux mails."

    system_instructions += ton
    system_instructions += user_indications


    if msg.elements:

        # fichiers Pdf ou txt seulement
        files = [file for file in msg.elements if "pdf" in file.mime or "text/plain" in file.mime]

        if not files:
            await cl.Message(content="Only PDF or Text file").send()
            return

        content = ""
        for file in files:
            content += "Fichier joint : "+file.name+"\n" 
            
            if "pdf" in file.mime:
                with open(file.path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    num_pages = len(pdf_reader.pages)

                    for page_num in range(num_pages):
                        page = pdf_reader.pages[page_num]
                        content += page.extract_text()
            
            if "text/plain" in file.mime:
                with open(file.path, "r") as f:
                    content += f.read()

            content += "\n"

        user_attached_files = f"""Vous devez répondre en prenant compte les [[pièces jointes]] aux mails.\nPièces jointes : {content}"""
    else:
        user_attached_files = ""

    system_instructions += user_attached_files
    print(system_instructions)
    runnable_model = setup_model()

    prompt = msg.content

    msg_res = cl.Message(content="")

    async for chunk in runnable_model.astream(
        {"sys_instruction": system_instructions, "prompt": prompt},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg_res.stream_token(chunk)

    await msg_res.send()