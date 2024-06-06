
  

mpnet-base-v2 bien meilleur que instructor pour fouiller les dépôts github

  
  
  
  

## PR_Reviewer

  

### 1. Introduction

Le **PR_Reviewer** permet d'appeler un modèle pour commenter les Pull Request effectuées sur Gituhb directement.

  

### 2. Fonctionnement

De par son utilisation de l'API Github, ce programme nécessite un fichier .env placé à son niveau, comportant les champs indiqués dans le .env.example, à savoir le nom du dépôt ciblé, son possesseur, et votre token Github pour avoir le droit d'utiliser l'API.
 
Le programme est divisé en 3 fonctions importantes.

```py
get_diff(pull_number)

generate_comment(diff_files)

post_comment(pull_number, comment)
```

**get_diff()** appelle l'API Github pour récupérer les fichiers contenus dans la Pull Request dont le numéro est en paramètre de la fonction.

**generate_comment()** s'occupe, en utilisant les fichiers récupérés précédemment, de générer une réponse en utilisant un modèle, choisi par la variable *llm_local*. Le modèle choisi ici est *llama3:instruct*, qui est une version légère mais performante avec des résultats assez bons sur ce genre de tâche. D'autres modèles ont été envisagés, notamment *codegemma* et *codegemma:7b-code*, sans grand succès ( résultats de [*codegemma*](https://github.com/eisenhowair/ProjetLLM/pull/33) et [*codegemma:7b-code*](https://github.com/eisenhowair/ProjetLLM/pull/34)). Le prompt choisi ici se veut assez simple, car sans nécessité de complexité. De ce fait, le modèle le respecte bien (parle en français, commente bien les différents fichiers, etc.).

**post_comment()** se charge ensuite d'envoyer le commentaire généré via l'API Github.

Ce programme fonctionne en utilisant un webhook, qu'il faut créer depuis Github directement (Settings -> Webhooks -> add webhook), Flask pour lancer un server local, et ngrok pour relier le webhook au server local.
Pour faire s'assembler le tout, voici les choses à faire:
- se créer un compte sur ngrok
- obtenir un token sur son compte
- installer ngrok sur votre machine puis le configurer avec les commandes suivantes: 
```bash
snap install ngrok
ngrok config add-authtoken # suivi de votre token ici
ngrok http 5000 # peut nécessiter un sudo
```
- après que l'écran noir de ngrok soit apparu, une url sera proposée à droite. C'est cette url, suivie de "/webhook", qu'il faudra entrer dans le webhook de Github
- lancer le programme python dans un autre terminal, l'autre terminal étant pris par ngrok

> [!WARNING]
> Attention, l'url de ngrok change quotidiennement pour la version gratuite, il faut donc penser à mettre à jour l'url du webhook sur Github à chaque fois avec la nouvelle url fournie par ngrok.


  
## Github_chatbot

### 1. Introduction

Le programme **github_chatbot_CL.py** utilise chainlit pour mettre en place un chatbot avec RAG portant sur un dépôt Github indiqué. Au lancement, il va vérifier si un index existe déjà à l'endroit indiqué (dans le dossier portant le nom du modèle d'embedding, lui-même dans le dossier vectorstores placé à la racine de ce projet), et le cas échéant le charge directement. L'opération prend quelques secondes, et des messages sur le chatbot permettent de voir si un index a été trouvé ou non. Si il n'y a pas d'index existant au préalable, cliquer sur les options (en bas à gauche de l'écran) permet d'entrer soit le nom du dépôt à récupérer et le nom de son possesseur, ou directement l'url complète. L'url toute seule suffit. Les documents du dépôt sont ensuite récupérés pour former un index, qui sera enregistré en local, pour ne pas avoir à le recréer à chaque fois. Dès lors, le chatbot permet de poser des questions au modèle sur le contenu de ce dépôt.

### 2. Fonctionnement

Le programme porte sur deux fichiers principalement, **github_chatbot_CL.py** et **github_recup.py**, ainsi qu'un troisième fichier **embedding_models.py** qui comporte les chemins des index, ainsi que les noms des modèles d'embedding. Comme pour **PR_Reviewer.py**, vous aurez besoin d'une variable d'environnement comportant votre token Github.

Le programme à lancer est **github_chatbot_CL.py**, avec la commande suivante :
```bash
chainlit run github_chatbot_CL.py
```
A noter les paramètres -h pour ne pas que chainlit ouvre de lui-même une nouvelle fenêtre, -w pour que chainlit s'actualise à chaque modification du code, et -d pour activer le debug.

  Lorsque l'utilisateur entre un dépôt github valide dans les paramètres, le décorateur chainlit *on_settings_update* va récupérer les valeurs, et les donner à la fonction *recup_index()*.  C'est dans cette fonction ,ainsi que *charge_index()* qui est appelée dès le début de *recup_index()* que le plus grand travail se fait. En effet, *charge_index()* va tout d'abord regarder si l'index existe déjà, et le récupérer si c'est le cas grâce aux différentes fonctions llama_index prévues pour l'occasion, ainsi que *from_persist_dir()*, de FAISS. Ces fonctions très simples d'utilisation permettent en quelques lignes de récupérer un index local, ainsi que le vectorstore.

  Si l'index n'existe pas, il est nécessaire d'en créer un. La première chose à faire est donc de récupérer les données du Github correspondant, et c'est à ça que sert la fonction *fetch_repository()*. Elle utilise le package 
```python
llama_index.readers.github
```
pour permettre de se connecter et récupérer les données voulues. Ces fonctions sont parmi les méthodes les plus maniables pour ce genre de tâche, et sont en plus claires dans leur manière d'utilisation. La méthode *GithubRepositoryReader()* en particulier est très intéressante. En plus des paramètres classiques, comme le nom du dépôt, son possesseur, ou le GithubClient pour la connexion, elle dispose de l'argument *filter_directories* qui prend un tuple de liste, cette dernière comportant des chemins du dépôt, et l'autre membre du tuple étant soit *GithubRepositoryReader.FilterType.EXCLUDE* soit *GithubRepositoryReader.FilterType.EXCLUDE*, suivant si vous voulez inclure ces chemins dans les documents à récupérer , ou les en exclure. Vient ensuite l'argument *filter_file_extensions* qui fonctionne avec le même principe, à la différence près que sa liste doit comporter des types de fichier (".pdf",".json", etc.) que vous pouvez là aussi choisir d'inclure ou d'exclure. Il est possible de spécifier les branches à récupérer, ou même les commit, tous deux dans les arguments de la fonction *load_data()*.

Les documents récupérés sont ensuite rendus à *charge_index()*, qui va se charger de créer l'index:
```python
d = 768 # dimensions du modèle d'embedding
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)
  storage_context_global = StorageContext.from_defaults(
  vector_store=vector_store)

index = VectorStoreIndex.from_documents(
  documents, storage_context=storage_context_global, show_progress=True)
index.storage_context.persist(index_path)

service_context = ServiceContext.from_defaults(
  embed_model=Settings.embed_model, llm=Settings.llm, callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))
```
Cettemanière de créer l'index est plus complexe que de faire quelque chose du type :
```python
vectorstore = FAISS.from_documents(
    documents=chunks, embedding=new_embeddings
)

vectorstore.save_local(new_index_path)
```
mais permet une plus grande liberté, surtout dans notre cas où l'on veut récupérer le *service_context*. Il est à noter que llama_index permet d'utiliser plusieurs types de VectorStore différents, FAISS n'en étant qu'un parmi d'autres. Après que l'index ait été crée, le programme retourne dans *recup_index()* pour la dernière partie, qui est la préparation de la *query_engine*. Il est possible de lui préciser le nombre maximal de documents dans lesquels elle peut piocher pour répondre aux questions de l'utilisateur avec l'argument *similarity_top_k*. 

Vient ensuite le moment de passer un prompt à la *query_engine*. Cette étape, même si non essentielle, car un prompt est déjà présent par défaut, permet d'améliorer les résultats par rapport aux attentes. Plusieurs méthodes existent pour ce faire. La première consiste à récupérer un prompt adapté depuis [langchain/hub](https://smith.langchain.com/hub), puis de le passer à un objet *LangchainPromptTemplate*, qu'on passera lui-même à la *query_engine*. On peut aussi passer un prompt fait maison à l'objet *LangchainPromptTemplate*, comme c'est le cas avec *create_prompt_simplifie()*. Attention au fait que llama_index et langchain n'ont pas les mêmes noms de variables pour les prompts, c'est pourquoi *template_var_mappings* est très important.

Enfin, il suffit de stocker la *query_engine* obtenue dans la session chainlit pour pouvoir l'appeler lorsqu'un message est écrit par l'utilisateur.

### 3. Remarques

Plusieurs choses à noter:
- Bien que le programme est censé supporter l'aspect connexion/profil proposé par chainlit (voir la fonction *auth_callback()* en commentaire), les réponses du modèle lorsque l'utilisateur est connecté à un profil ne s'envoient pas, alors qu'elles sont bien générées, c'est pourquoi la fonction de connexion est en commentaire.
- Pour changer le modèle d'embedding à utiliser, mettre son nom dans l'argument index_path de la fonction *charge_index()*, et le chemin de son index quelques lignes plus haut dans
```python
  embed_model = HuggingFaceEmbedding(
    # ici changer le modèle selon embedding_models
    model_name=embedding_model_hf_en_mpnet)
```
Il existe déjà des variables correspondantes pour différents modèles dans le fichier **embedding_models.py**
- L'objet Settings sert de variable globale utilisée par llama_index
- Même si un message d'erreur par exemple _Error while flushing create_step_ apparait, la réponse du modèle devrait quand même apparaitre sur chainlit. C'est uniquement si le temps de réponse est apparu sur le terminal alors que chainlit indique la réponse est encore en cours de chargement qu'il y a eu un problème
- Si toutefois vous souhaitez tout de même essayer avec le système de profil, notamment pour pouvoir mieux suivre les différentes conversatins avec un plus grand niveau de détail, il vous faudra décommenter la fonction *auth_callback()*, créer un compte [Literal AI](https://cloud.getliteral.ai/), récupérer le token associé (dans Settings -> General -> default key), et le mettre dans le fichier d'environnement .env, avec le secret chainlit que vous pouvez générer de cette manière :
```bash
chainlit create-secret
```
  Se fier au .env.example pour le noms des variables à respecter.
- La fonction *ask_github_index()*, ainsi que les quelques lignes commentées à la fin du fichier **github_recup.py** sont là pour tester le fichier tout seul sans avoir recours à chainlit.
- Par défaut, le modèle d'embedding est "sentence-transformers/all-mpnet-base-v2", car jugé plus efficace par rapport au code que même "hkunlp/instructor-large", qui pourtant a eu de très bons résultats tout le long de ce projet. Cela dit, lui aussi s'appuye fortement sur les fichiers textes pour émettre ses avis sur les dépôts Github, le rendant moins utile.

> [!WARNING]
> Attention, si vous utilisez des variables provenant de **embeddings_model.py**, il est nécessaire de mettre l'import du fichier après la ligne modifiant le path, elle même après l'import sys.
> Les formatteurs automatiques vont déranger cet ordre, empêchant l'accès aux variables contenant les noms des modèles d'embedding, ainsi que le chemin de leur index.

## Sources

https://blog.gopenai.com/building-a-rag-chatbot-using-llamaindex-groq-with-llama3-chainlit-b1709f770f55 (chainlit avec llama index)

https://ogre51.medium.com/context-window-of-language-models-a530ffa49989 (context window)

https://docs.chainlit.io/integrations/llama-index ( documentation chainlit)

https://docs.llamaindex.ai/en/stable/examples/prompts/prompts_rag/ (prompt llama_index)

https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/ (doc officielle)

https://github.com/joaomdmoura/crewAI
https://docs.crewai.com/tools/GitHubSearchTool (impossible d'importer crewAI)

https://lightning.ai/lightning-ai/studios/chat-with-your-code-using-rag (a donné des pistes)

https://docs.llamaindex.ai/en/latest/module_guides/indexing/vector_store_index/ (pour utiliser le VectorStoreIndex avec llama_index)
