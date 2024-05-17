from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from operator import itemgetter

model = Ollama(base_url="http://localhost:11434", model="llama3:instruct")

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
                "Tu parles uniquement français. Ton rôle est de créer un seul exercice de mathématiques auquel l'utilisateur doit trouver la réponse,\
            en te basant sur un ou plusieurs intérêts suivants : " + loisirs + ". L'exercice doit impliquer :{question}, et être du niveau d'un élève ayant {niveau_scolaire}."
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
    # cl.user_session.set("runnable", runnable_exercice)
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
                    "Tu es un maitre d'école avec des élèves de {niveau_scolaire} français .Tu dois aider l'utilisateur \
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
    # cl.user_session.set("runnable", runnable_corrige)
    if cl.user_session.get("compris") == True:
        cl.user_session.set("dernier_exo", "")

    return runnable_corrige
