import chainlit as cl
from chainlit.input_widget import TextInput,Select,Switch,Slider


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
        ]
    ).send()
    print(settings["AgentName"])
    print(settings["Model"])
    print(settings["Streaming"])

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