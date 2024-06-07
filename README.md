
## Procédure de lancement

- lancer ```bash curl -fsSL https://ollama.com/install.sh | sh ``` pour installer ollama
- puis ```bash ollama run llama3:8b ``` pour installer le modèle
- http://localhost:11434/ permet de vérifier le lancement du server

```python
   model = Ollama(   
        base_url='http://localhost:11434',
        model="llama3:8b")
```

permet d'utiliser l'API REST de Ollama installé localement.

Il est nécessaire d'installer les dépendances mentionnées dans le fichier **requirements.txt** à la racine de ce projet. De plus, certaines applications ont leur propre **requirements.txt** qu'il convient de respecter aussi. La commande est la suivante :
```python
pip install -r requirements.txt
```

Enfin, chaque dossier disposant d'un fichier **.env.example** nécessite un fichier **.env**, dont le contenu sera indiqué dans le **.env.example**, et dans le **README.md** associé.

## Applications
> [!IMPORTANT]
> mettre ici le tableau

## Remarques

Tous les programmes bénéficiant d'une interface de connexion ont "elias" comme nom d'utilisateur et mot de passe. Ces valeurs sont changeables dans la fonction `auth_callback` du programme lancé.

Attention car si un programme utilisant un index a un problème pendant la création dudit index, le dossier contenant l'index aura été crée, mais vide. De ce fait, le programme, lors de sa prochaine exécution, va penser que l'index existe car le dossier existe, ce qui n'est pas le cas. Il renverra donc une erreur de type:
```bash
ValueError: No existing llama_index.vector_stores.faiss.base found at vectorstores/llama_index_mpnet/default__vector_store.json.
```
Pour éviter cette erreur, il est généralement plus simple de supprimer le dossier vide.

## Sources

https://github.com/ollama/ollama/blob/main/docs/tutorials/langchainpy.md

https://github.com/sudarshan-koirala/langchain-ollama-chainlit/blob/main/simple_chaiui.py

https://medium.com/@tahreemrasul/building-a-chatbot-application-with-chainlit-and-langchain-3e86da0099a6 ( pour utiliser la ConversationBufferMemory)

#### Llama2 Chat Integration (`llama2-chat.py`)

- `load_llama`: Loads the Llama2 model from HuggingFace with the specified tokenizer and streamer.
- `main` (decorated with `@cl.on_chat_start`): Initializes the LLM chain with a prompt for answering questions.
- `run`: Processes incoming messages and provides responses using the LLM chain.

- dans terminal:
    huggingface-cli login
    sur machine de la fac:
    /home/UHA/e2303253/.local/bin/huggingface-cli login


- mettre le token généré depuis le profil huggingface


- ollama run llama3:8b
