from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from chainlit.input_widget import Slider
from operator import itemgetter


from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

from chainlit.types import ThreadDict

model = Ollama(base_url="http://localhost:11434", model="llama3:instruct")


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("elias", "elias"):
        return cl.User(
            identifier="Elias", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():
    await cl.ChatSettings(
        [
            Slider(
                id="age_cible",
                label="Age niveau exercice",
                initial=8,
                min=3,
                max=22,
                step=1,
                tooltip="en années",
            ),
        ]
    ).send()

    loisirs = await cl.AskUserMessage(content="Quels sont vos centres d'intérêt?", author="Aide", timeout=3000).send()
    # print(loisirs["output"])
    cl.user_session.set("loisirs", loisirs["output"])
    response = "Merci! Quel genre d'exercice voulez-vous?"
    await cl.Message(content=response, author="Aide").send()

    cl.user_session.set("compris", True)
    cl.user_session.set("tentatives", 0)
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))



async def verifie_comprehension():

    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    res = await cl.AskActionMessage(
        content="Avez-vous compris?",
        actions=[
            cl.Action(name="continue", value="compris", label="✅ Compris"),
            cl.Action(name="cancel", value="pas_compris",
                      label="❌ Pas compris"),
        ],
        disable_feedback=True,
        author="Correcteur",
        timeout=3000
    ).send()



    if res.get("value") == "pas_compris":
        cl.user_session.set("compris", False)
        cl.user_session.set("tentatives", cl.user_session.get("tentatives")+1)

        msg = await cl.Message(
            content="Qu'avez-vous pas compris?",
        ).send()
    else:
        cl.user_session.set("compris", True)
        cl.user_session.set("tentatives", 1)

        msg = await cl.Message(
            content="Félicitations! Quel autre exercice voulez-vous?",
        ).send()
    
    # on met à jour l'historique
    memory.chat_memory.add_ai_message("Avez-vous compris?")
    memory.chat_memory.add_user_message(res.get("value"))
    memory.chat_memory.add_ai_message(msg.content)


@cl.step(type="run", name="runnable_generation")
def setup_exercice_model():
    """
    Configure le prompt et le Runnable pour générer des exercices de mathématiques personnalisés en fonction des centres d'intérêt de l'utilisateur.

    Returns:
        None : Met à jour la session utilisateur avec le nouveau Runnable pour générer des exercices.
    """

    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    loisirs = cl.user_session.get("loisirs")
    prompt_exercice = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu parles uniquement français. Ton rôle est de créer un seul exercice de mathématiques \
            en te basant sur un ou plusieurs intérêts suivants : " + loisirs + ". L'exercice doit impliquer :{question}, et être du niveau d'un élève ayant {niveau_scolaire}"
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ]
    )

    runnable_exercice = (
        RunnablePassthrough.assign(
            history=RunnableLambda(
                memory.load_memory_variables) | itemgetter("history")
        ) 
        | prompt_exercice
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable_exercice)
    return runnable_exercice


@cl.step(type="run", name="runnable_corrige")
def setup_corrige_model():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    if cl.user_session.get("tentatives") < 3:
        print("partie aide d'exercice")
        print("Nombre de tentatives faites: " +
              str(cl.user_session.get("tentatives")))
        prompt_corrige = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Tu es un maitre d'école avec des enfants de {niveau_scolaire} français .Tu dois aider l'utilisateur \
                    à résoudre l'exercice de mathématiques suivant: {dernier_exo}. \
                Si la réponse de l'utilisateur n'est pas correcte, donne un indice utile pour aider l'utilisateur à trouver la solution. \
                S'il répond correctement, félicite-le. Tu ne dois jamais donner la réponse toi-même."
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ]
        )
    else:
        print("partie correction d'exercice")
        prompt_corrige = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Tu parles uniquement français. Ton rôle est de corriger l'exercice de mathématiques suivant: {dernier_exo}. \
                Si la réponse {question} donnée par l'utilisateur est juste, félicite-le. \
                Sinon, dis-lui qu'il a faux, et dis-lui la correction de l'exercice."
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ]
        )

    runnable_corrige = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            )
            | prompt_corrige
            | model 
            | StrOutputParser())
    cl.user_session.set("runnable", runnable_corrige)
    if cl.user_session.get("compris") == True:
        cl.user_session.set("dernier_exo", "")

    return runnable_corrige


@cl.on_message
async def on_message(message: cl.Message):
    """
    Callback fonction appelée à chaque réception d'un message de l'utilisateur.
    Gère la logique principale de la conversation en fonction de l'état de la session utilisateur.

    Args:
        message (cl.Message): Le message envoyé par l'utilisateur.

    Returns:
        None : Envoie une réponse appropriée à l'utilisateur en fonction du contexte de la conversation.
    """
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    if cl.user_session.get("age_niveau"):
        niveau_scolaire = str(cl.user_session.get("age_niveau"))+" ans"
    else:
        niveau_scolaire = "5 ans"

    if cl.user_session.get("compris") == True:  # partie génération d'exercice
        dernier_exo = ""
        print("partie génération d'exercice")
        runnable = setup_exercice_model()

        msg = cl.Message(content="", author="Générateur")
        async for chunk in runnable.astream(
            {"question": message.content, "niveau_scolaire": niveau_scolaire},
            config=RunnableConfig(
                callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
            dernier_exo += chunk
        cl.user_session.set("dernier_exo", dernier_exo)
        cl.user_session.set("compris", False)
        await msg.send()

        memory.chat_memory.add_user_message(message.content)
        memory.chat_memory.add_ai_message(msg.content)

        # partie correction d'exercice
    elif cl.user_session.get("compris") == False:
        runnable = setup_corrige_model()

        msg = cl.Message(content="", author="Correcteur")
        async for chunk in runnable.astream(
            {"question": message.content,
                "dernier_exo": cl.user_session.get("dernier_exo"),
                "niveau_scolaire": niveau_scolaire
             },
            config=RunnableConfig(
                callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)
        print("msg:"+str(msg.content))
        await msg.send()
        memory.chat_memory.add_user_message(message.content)
        memory.chat_memory.add_ai_message(msg.content)
        await verifie_comprehension()


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("age_niveau", settings['age_cible'])


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

    await cl.ChatSettings(
        [
            Slider(
                id="age_cible",
                label="Age niveau exercice",
                initial=8,
                min=3,
                max=22,
                step=1,
                tooltip="en années",
            ),
        ]
    ).send()

    setup_exercice_model()
