from operator import itemgetter
from langchain_community.llms import Ollama

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

from chainlit.types import ThreadDict
import chainlit as cl


def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = Ollama(model="llama3:8b")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


@cl.password_auth_callback
def auth():
    return cl.User(identifier="test")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)

"""import chainlit as cl
from chainlit.input_widget import TextInput,Select,Switch,Slider

from typing import Optional


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="GPT-3.5",
            markdown_description="The underlying LLM model is **GPT-3.5**.",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="GPT-4",
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://picsum.photos/250",
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"starting chat using the {chat_profile} chat profile"
    ).send()



@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_history", [])
    cl.user_session.set("chat_history", [{"role": "system",
                        "content": "behave as if you are news reporter."}])

@cl.on_message
async def generate_response(query: cl.Message):
    chat_history = cl.user_session.get("chat_history")
    chat_history.append({"role": "user", "content": query.content})
    
    response = cl.Message(content="")
    answer = 'ollama.chat(model="llama3:8b", messages=chat_history, stream=True)'
    
    complete_answer = ""
    for token_dict in answer:
        token = token_dict["message"]["content"]
        complete_answer += token
        await response.stream_token(token)
    
    chat_history.append({"role": "assistant", "content": complete_answer})    
    cl.user_session.set("chat_history", chat_history)

    await response.send()

@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            TextInput(id="AgentName", label="Agent Name", initial="AI"),
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
                initial_index=0,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="SAI_Height",
                label="Stability AI - Image Height",
                initial=512,
                min=256,
                max=2048,
                step=64,
                tooltip="Measured in pixels",
            ),
        ]
    ).send()


    res = await cl.AskActionMessage(
        content="Pick an action!",
        actions=[
            cl.Action(name="continue", value="continue", label="✅ Continue"),
            cl.Action(name="cancel", value="cancel", label="❌ Cancel"),
        ],
    ).send()

    if res and res.get("value") == "continue":
        await cl.Message(
            content="Continue!",
        ).send()

    res = await cl.AskUserMessage(content="What is your name?", timeout=10).send()
    if res:
        await cl.Message(
            content=f"Your name is: {res['output']}",
        ).send()
    text_content = "Hello, this is a text element."
    elements = [
        cl.Text(name="simple_text", content=text_content, display="inline")
    ]

    await cl.Message(
        content="Check out this text element!",
        elements=elements,
    ).send()

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)"""