from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    model = Ollama(base_url="http://localhost:11434", model="llama3")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu parles en français. A chaque échange, tu vas recevoir l'historique des précédents messages pour que tu aies le contexte de la converssation mais garde le simplement comme historique, ne l'écrit pas."
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    with open('historique.txt', 'r') as f:
        contenu = f.read()
        print("contenu=",contenu)

    with open('historique.txt', 'a') as ff:
        ff.write(message.content)

        ff.write('\n')

    allMessages = contenu+message.content


    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": "Historique : "+contenu+"\nNouveau contexte :  ["+message.content+"]"},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
