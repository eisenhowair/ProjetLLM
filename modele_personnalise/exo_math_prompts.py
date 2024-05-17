from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from chainlit.input_widget import TextInput,Select,Switch,Slider


model = Ollama(base_url="http://localhost:11434", model="llama3:instruct", mirostat=1,mirostat_eta = 2)


@cl.on_chat_start
async def on_chat_start():
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
        author="Correcteur",
        timeout=3000
    ).send()
    if res.get("value") == "pas_compris":
        cl.user_session.set("compris", False)
        cl.user_session.set("tentatives", cl.user_session.get("tentatives")+1)

        await cl.Message(
            content="Qu'avez-vous pas compris?",
        ).send()
    else:
        cl.user_session.set("compris", True)
        cl.user_session.set("tentatives", 1)

        await cl.Message(
            content="Félicitations! Quel autre exercice voulez-vous?",
        ).send()

@cl.step(type="runnable",name="runnable_generation")
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
                "Tu parles uniquement français. Ton rôle est de créer un seul exercice de mathématiques \
            en te basant sur un ou plusieurs intérêts suivants : " + loisirs + ". L'exercice doit impliquer :{question}, et être du niveau d'un élève ayant {niveau_scolaire}"
            ),
            ("human", "{question}")
        ]
    )

    runnable_exercice = prompt_exercice | model | StrOutputParser()
    cl.user_session.set("runnable", runnable_exercice)
    return runnable_exercice

@cl.step(type="runnable",name="runnable_corrige")
def setup_corrige_model(indice_precedent = ""):
    if cl.user_session.get("tentatives") < 3:
        print("partie aide d'exercice")
        print("Nombre de tentatives faites: "+str(cl.user_session.get("tentatives")))
        prompt_corrige = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Tu es un maitre d'école avec des enfants de {niveau_scolaire} français .Tu dois aider l'utilisateur \
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

    if cl.user_session.get("age_niveau"):
        niveau_scolaire= str(cl.user_session.get("age_niveau"))+" ans"
    else:
        niveau_scolaire = "5 ans"

    if cl.user_session.get("compris") == True:  # partie génération d'exercice
        dernier_exo = ""
        print("partie génération d'exercice")
        runnable = setup_exercice_model()
        #runnable = cl.user_session.get("runnable")

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
        runnable = setup_corrige_model()
        #runnable = cl.user_session.get("runnable")  # type: Runnable

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



@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("age_niveau",settings['age_cible'])
    print("on_settings_update", settings)