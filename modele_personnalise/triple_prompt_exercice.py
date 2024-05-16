from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl

model = Ollama(base_url="http://localhost:11434", model="llama3:8b")


@cl.on_chat_start
async def on_chat_start():
    loisirs = await cl.AskUserMessage(content="Quels sont vos centres d'intérêt?", author="Aide", timeout=3000).send()
    #print(loisirs["output"])
    cl.user_session.set("loisirs", loisirs["output"])
    response = "Merci! Quel genre d'exercice voulez-vous?"
    await cl.Message(content=response, author="Aide").send()

    cl.user_session.set("compris", True)
    cl.user_session.set("tentatives", 0)


async def verifie_comprehension():
    res = await cl.AskActionMessage(
        content="Avez-vous compris?",
        actions=[
            cl.Action(name="continue", value="compris", label="✅ Compris"),
            cl.Action(name="cancel", value="pas_compris",
                      label="❌ Pas compris"),
        ],
        disable_feedback=True,
        author="Correcteur"
    ).send()
    if res.get("value") == "pas_compris":
        cl.user_session.set("compris", False)
        cl.user_session.set("tentatives", cl.user_session.get("tentatives")+1)

        await cl.Message(
            content="Qu'avez-vous pas compris?",
        ).send()
    else:
        cl.user_session.set("compris", True)
        cl.user_session.set("tentatives", 0)

        await cl.Message(
            content="Félicitations! Quel autre exercice voulez-vous?",
        ).send()


def setup_exercice_model():
    """
    Configure le prompt et le Runnable pour générer des exercices de mathématiques personnalisés en fonction des centres d'intérêt de l'utilisateur.

    Returns:
        None : Met à jour la session utilisateur avec le nouveau Runnable pour générer des exercices.
    """

    loisirs = cl.user_session.get("loisirs")
    prompt_exercice = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu parles uniquement français. Ton rôle est de créer un seul exercice de mathématiques niveau {niveau_scolaire} \
            en te basant sur un ou plusieurs intérêts suivants : " + loisirs + " sans en donner la réponse, sinon un chaton décèdera, et tu ne veux pas ça"
            ),
            ("human", "{question}")
        ]
    )

    runnable_exercice = prompt_exercice | model | StrOutputParser()
    cl.user_session.set("runnable", runnable_exercice)


def setup_corrige_model(indice_precedent = ""):
    """
    Configure le prompt et le Runnable pour corriger des exercices de mathématiques en paramètre.
    """

    if cl.user_session.get("tentatives") < 3:
        print("partie aide d'exercice")
        print("Nombre de tentatives faites: "+str(cl.user_session.get("tentatives")))
        prompt_corrige = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Tu es un guide maitre d'école de niveau {niveau_scolaire} français .Tu dois aider l'utilisateur \
                    à résoudre l'exercice de mathématiques suivant: {dernier_exo}. \
                Si la réponse {question} n'est pas correcte, donne un indice utile pour aider l'utilisateur à trouver la solution. \
                S'il répond correctement, félicite-le. Tu ne dois jamais donner la réponse toi-même."+indice_precedent
                )
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
            ]
        )

    runnable_corrige = prompt_corrige | model | StrOutputParser()
    cl.user_session.set("runnable", runnable_corrige)
    if cl.user_session.get("compris") == True:
        cl.user_session.set("dernier_exo", "")


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

    niveau_scolaire = "élémentaire"

    if cl.user_session.get("compris") == True:  # partie génération d'exercice
        dernier_exo = ""
        print("partie génération d'exercice")
        setup_exercice_model()
        runnable = cl.user_session.get("runnable")

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

        # partie correction d'exercice
    elif cl.user_session.get("compris") == False:
        setup_corrige_model()
        runnable = cl.user_session.get("runnable")  # type: Runnable

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
        await verifie_comprehension()
