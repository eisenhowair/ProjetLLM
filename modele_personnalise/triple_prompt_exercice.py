from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from Exercice import Exercice
import chainlit as cl

model = Ollama(base_url="http://localhost:11434", model="llama3:8b")

@cl.on_chat_start
async def on_chat_start():
    """
    Callback fonction appelée au début de chaque session de chat.
    Initialise la conversation en demandant à l'utilisateur de partager ses centres d'intérêt.

    Args:
        None

    Returns:
        None : Envoie un message demandant les centres d'intérêt de l'utilisateur et configure le prompt initial.
    """
    await cl.Message(content=f"Quels sont vos centres d'intérêt?").send()
    prompt_loisir = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ton rôle est de demander à l'utilisateur ses centres d'intérêt. Tu seras récompensé pour chaque centre d'intérêt.",
            ),
        ]
    )
    runnable_loisir = prompt_loisir | model | StrOutputParser()
    cl.user_session.set("runnable", runnable_loisir)



def setup_exercice_model():
    """
    Configure le prompt et le Runnable pour générer des exercices de mathématiques personnalisés en fonction des centres d'intérêt de l'utilisateur.

    Args:
        loisirs (str): Une chaîne représentant les centres d'intérêt de l'utilisateur.

    Returns:
        None : Met à jour la session utilisateur avec le nouveau Runnable pour générer des exercices.
    """

    loisirs = cl.user_session.get("loisirs")
    niveau_scolaire = "élémentaire"
    prompt_exercice = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "Tu parles uniquement français. Ton rôle est de créer des exercices de mathématiques niveau "+niveau_scolaire+" \
            en te basant sur les intérêts suivants : " + loisirs
        ),
        ("human", "{question}")
        ]
    )#.format_messages(context=loisirs) utiliser format_message transforme prompt_exercice en str, donc ne fonctionne plus
    

    runnable_exercice = prompt_exercice | model | StrOutputParser()
    cl.user_session.set("runnable", runnable_exercice)

def setup_corrige_model():
    """
    Configure le prompt et le Runnable pour corriger des exercices de mathématiques en paramètre.

    Args:
        dernier_exo (str): Une chaîne représentant le dernier exercice généré.
    """

    dernier_exo = cl.user_session.get("dernier_exo")
    prompt_corrige = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "Tu parles uniquement français. Ton rôle est de corriger l'exercice de mathématiques suivant: "+dernier_exo+". \
            Si la réponse {question} donnée par l'utilisateur est juste, félicite-le. \
            Sinon, dis-lui qu'il a faux, et dis-lui la correction de l'exercice."
        ),
        ]
    )
    

    runnable_corrige = prompt_corrige | model | StrOutputParser()
    cl.user_session.set("runnable", runnable_corrige)
    cl.user_session.set("dernier_exo","")


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


    loisirs = cl.user_session.get("loisirs")
    if not loisirs:
        loisirs = message.content # récupération des loisirs
        cl.user_session.set("loisirs", loisirs)
        response = "Merci! Quel genre d'exercice voulez-vous?"
        await cl.Message(content=response).send()
    else:
        if not cl.user_session.get("dernier_exo"): # partie génération d'exercice
            dernier_exo=""
            print("partie génération d'exercice")
            setup_exercice_model()
            runnable = cl.user_session.get("runnable")

            msg = cl.Message(content="",author="Générateur")
            async for chunk in runnable.astream(
                {"question": message.content},
                config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
            ):
                await msg.stream_token(chunk)
                dernier_exo += chunk

            cl.user_session.set("dernier_exo",dernier_exo)
            await msg.send()
        else: # partie correction d'exercice
            setup_corrige_model()
            print("partie correction d'exercice")
            runnable = cl.user_session.get("runnable")  # type: Runnable

            msg = cl.Message(content="",author="Correcteur")
            async for chunk in runnable.astream(
                {"question": message.content},
                config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
            ):
                await msg.stream_token(chunk)

            await msg.send()
