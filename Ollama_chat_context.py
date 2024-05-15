from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    model = Ollama(base_url="http://localhost:11434", model="llama3")

    memory = ConversationBufferMemory(memory_key="history", input_key="input", max_token_limit=1000)
    # memory.chat_memory.add_user_message("Bonjour je suis Etudiant")

    prompt =  PromptTemplate(
        input_variables=['history', 'input'],
        template="""
        Tu es un bot de conversation français. Un utilisateur va communiquer avec toi. Maintiens un ton formel dans tes réponses.
        Historique de conversation :
        {history}

        Humain : {input}
        AI :
        """
    )
   
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=model)
    cl.user_session.set("conversation", conversation)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("conversation")

    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        print(chunk)
        await msg.stream_token(chunk["response"])

    await msg.send()