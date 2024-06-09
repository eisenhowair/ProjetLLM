<a name="readme-top"></a>
# RAG_moodle_ENT.py

## Introduction 

Le programme nécessite les dépendances du fichier **requirements.txt** à la racine de ce dépôt, ainsi que celles du fichier **requirements.txt** dans ce dossier.
De plus, avoir installé ollama (par exemple via le fichier **setup.sh**), et récupéré le modèle llama3:instruct est nécessaire.
```py
pip install -r ../requirements.txt
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3:instruct
```


1st version : Question par console fonctionne bien en utilisant un index, mais ne fonctionne pas avec chainlit car n'utilise pas l'index.

2nd version : Le programme fonctionne maintenant avec chainlit, et ne met que 3 minutes à répondre aux questions, en se basant sur un index composé de plusieurs fichiers de taille et type variable. Le modèle d'embedding n'est plus nomic-embed-text, qui faisait trop d'erreur dans sa sélection de documents, mais aussi de chunks, mais instructor-large, qui est bien plus consistant. L'index est stocké localement pour ne pas avoir à le recréer à chaque exécution du programme.

A été ajouté un système de connexion permettant de consulter les anciennes conversations, ainsi que de suivre l'exécution des requêtes via literal.ai (en ayant un fichier .env correspondant aux attentes demandées dans le chapitre suivant)

3rd version (latest) : Le programme a maintenant une mémoire lui permettant de se souvenir des messages d'une conversation, permettant d'avoir une véritable discussion. En plus de ça, l'index se génère en se basant sur plusieurs page web, ainsi que les différents cours disponibles sur moodle, en utilisant BeautifulSoup et Selenium. Les différents fichiers sur moodle sont téléchargés dans le dossier differents_textes/moodle (attention car il utilise un chemin absolu, voir **webscraper.py**). Il est aussi possible de choisir parmi plusieurs modèles d'embedding dans les options chainlit mises en place.

## Fonctionnalités plus précises

Ce programme dispose non seulement d'un historique des conversations permettant au modèle d'avoir accès à tous les messages échangés, mais en plus de quitter la conversation, et y revenir plus tard grâce à la fonction *on_chat_resume()*. Pour cela est nécessaire la fonction *auth_callback()* permettant de se connecter, ainsi qu'un fichier .env. Dans ce fichier devront être les variables LITERAL_API_KEY et CHAINLIT_AUTH_SECRET. Pour cela, créer un compte [Literal AI](https://cloud.getliteral.ai/), récupérer le token associé (dans Settings -> General -> default key), et le mettre dans le fichier d'environnement .env, avec le secret chainlit que vous pouvez générer de cette manière :
```bash
chainlit create-secret
```

Au lancement du programme, la fonction *charge_index()* est appelée pour vérifier la présence d'un index correspondant au modèle d'embedding utilisé. S'il existe, on utilise FAISS pour le récupérer. Sinon, on le crée à partir de 2 choses:
- un dictionnaire d'url (webpage_dict)
- un répertoire de fichiers (differents_textes)
Les 2 sont modifiables. Il est tout à fait possible de rajouter ou enlever des url, ainsi que des fichiers dans le dossier, voire même modifier le dossier cible (attention au chemin dans ce cas). Cela étant, l'index ne peut pas se modifier, il faudra donc le générer à nouveau, en supprimant le dossier où il est stocké (très probablement dans vectorstores/..)

### Webscraping

La création de chunks à partir d'url se fait en utilisant BeautifulSoup et Selenium, dans le fichier **web_scraper.py**. La fonction *load_web_documents_firefox()* se lance avec les urls en paramètre. Elle commence par utiliser son second paramètre, qui est une url un peu spéciale ,car c'est celle qui permet de se connecter à l'UHA. Il est nécessaire de se connecter à l'UHA pour pouvoir accéder à ses pages comme celles de moodle. Le driver crée par Selenium y est donc envoyé pour se connecter, en utilisant comme valeurs de connexion celles dans le fichier .env, qui doit avoir été préalablement rempli avec une adresse mail UHA et son mot de passe associé. Le fichier .env nécessaire est celui dans le dossier sandboxRAG. 

Après s'être connecté avec succès, le driver peut passer par toutes les urls du dictionnaire. Pour chaque url, on regarde d'abord si elle mène à un fichier pdf, auquel cas l'url est stockée dans une liste à part. Sinon, suivant le type de l'url, qui est donné lors de la création du dictionnaire (pour les urls initiales), différents traitement prennent place.

Par exemple, si on a affaire au type *webpage_from_moodle*, il faut rajouter *&redirect=1* à l'url avant de s'y rendre, car c'est comme ça que fonctionnent les liens de page web externes mis dans moodle.

Si l'url est de type *accueil_moodle*, un traitement spécifique, car on veut récupérer les urls de toutes les UE présentes sur la page moodle, et leur donner un type spécifique. Pour toutes les urls avec ce type, soit toutes les UE sur moodle, on appelle la fonction *extract_moodle_links()* qui va récupérer toutes les urls présente sur le moodle de l'UE en question.

Puis toutes les urls récupérées, qui n'étaient donc pas dans le dictionnaire de base, sont rajoutés au dictionnaire pour qu'il y passe plus tard.

Enfin pour toutes les pages, le code source est récupérée via le driver Selenium,et ajouté à une liste après avoir été retravaillé, tout comme les pdfs récupérés au début de la boucle. Faire tous les pdfs à la fin du programme limite le nombre de problème qui avait lieu notamment parce que le driver ne savait pas quoi faire lorsqu'un onglet s'ouvrait pour les pdfs, et donc buguait.
Même comme ça, le téléchargement du premier pdf pose souvent problème, forçant l'utilisateur a actualisé la page lui-même. L'opération prend 3-4 minutes.

> [!IMPORTANT]
> L'endroit où les pdfs sont téléchargés est indiqué dans la fonction *prepare_options()*. Le chemin étant absolu, il est impératif de le changer avant utilisation.
> Selon la version de chainlit utilisée, la ligne *allow_dangerous_deserialization=True,* peut créer une erreur. Si c'est le cas, la commenter ou l'enlever règlera le problème.

### Indexation

De retour dans *charge_index()*, on récupère les chunks des page web, puis on lance *load_new_documents()* avec comme paramètre le chemin du dossier où sont téléchargés les fichiers pdf. Cette fonction se chargera de transformer ces fichiers en chunks. L'overlap du *RecursiveCharacterTextSplitter()* est assez haut, pour essayer de garder un maximum de contexte.

Une fois ces chunks générés, on les ajoute à ceux obtenus précédemment avec le contenu des pages web, puis on crée le vectorstore à partir de ces chunks, avec FAISS.
Il ne reste alors plus qu'à enregistrer localement ce vectorstore dans un dossier pour ne pas avoir à le crér à chaque lancement du programme, et à l'enregistrer dans la session chainlit sous forme de retriever pour gagner du temps à chaque appel.

### Utilisation

Dès lors que l'utilisateur envoie un message, le programme appelle *setup_model()* pour préparer le Runnable qui permettra d'y répondre. Ce Runnable est composé du modèle utilisé pour la réponse, soit **llama3:instruct**, et du prompt:
```python
    prompt_exercice = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Instruction: Répondre en francais à la question de l'utilisateur en te basant **uniquement** sur le contexte suivant fourni.
                Si tu ne trouves pas la réponse dans le contexte, demande à l'utilisateur d'être plus précis au lieu de deviner.
                Context:{context}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Question: {question}"),
            ("ai", """Réponse:"""),
        ]
    )
```
Ce type de prompt, sous la forme d'un dialogue, est ce qui s'est révélé le plus efficace des différents essais réalisés. On lui passe les anciens messages pour que le modèle ait la conversation entière en tête pour améliorer ses réponses en cas de réelle conversation, et le tour est joué.

La dernière étape consiste à trouver depuis le vectorstore de quoi alimenter la variable {context}, et c'est là que *trouve_contexte()* intervient. Cette fonction utilise le retriever qui est dans la session pour faire une *similarity_search()* récupérant depuis 3 documents maximum (paramètre évidemment modifiable). Après avoir organisé les chunks récupérés pour que le modèle puisse y voir plus clair, on return le context formé ainsi.

### Changement de modèle

Changer de modèle via les options appelle la fonction *charge_index()*, et créera donc un nouvel index, si celui correspondant à ce modèle n'est pas déjà existant.


# RAGPdfOrTxt.py

Il s'agit d'une version plus primitive de RAG avec FAISS, fonctionnelle, quoiqu'un peu plus lente. Le RAG ici se fait fichier par fichier, en un ajoutant l'un après l'autre.
La différence proposée par ce programme est l'utilisation d'une *RetrievalQAChain*, qui prend un peu plus de temps.  L'objet *rag_chain* est légèrement plus rapide, mais semble plus compliquée à faire fonctionner avec chainlit, d'où la présence d'un appel manuel (en commentaire) pour tester.


# Llama_index_HF.py

Ce programme permet de tester llama_index sans chainlit. Il a servi de base aux autres programmes utilisant llama_index.

# Autres

## differents_textes/

Comporte plusieurs textes de longueur et type différent, pour vérifier le bon fonctionnement des indexs. C'est aussi dans ce dossier que les pdfs sont téléchargés depuis moodle, dans le dossier differents_textes/moodle.

## brouillonRAG/

Un dossier avec des programmes plus vieux, servant à tester le fonctionnement des différentes techonologies le long de ce projet. Certains programmes proviennent de Github ou de guides internet.

## utils/

- **embedding_models.py** comporte le nom de plusieurs modèles d'embedding différents, ainsi que le chemin de leur index respectif. Ce fichier est utilisé dans plusieurs applications de ce projet.
- **web_scraper.py** sert à récupérer les données des sites web en utilisant Selenium. Il est utilisé dans le fichier **RAG_moodle_ENT.py** mentionné plus haut.
- **manip_documents.py** est utilisé lui aussi dans **RAG_moodle_ENT.py**, et comporte les fonctions traitant les fichiers, pour rendre **RAG_moodle_ENT.py** plus lisible.
- **modify_pdf_metadata** permet de "réparer" des fichiers pdfs auxquels il manquerait des métadata, ce qui pourrait affecter la qualité du RAG. Les métadata sont à mettre manuellement, avant de lancer le programme. Des exemples d'utilisation sont compris dans le fichier.
- **compare_embedding_model.py** compare 5 modèles d'embedding, en posant une question, et regardant à quel point les modèles regardent dans les bons chunks pour trouver les réponses. Un graphique est ensuite affiché. Ce programme nécessite donc que les 5 modèles d'embedding aient un index, et peut donc nécessiter un certain temps si ce n'est pas le cas, et la quantité de documents à vectoriser est non négligeable.

## screenshot_compare_embedding/

Dans ce dossier vont tous les graphiques issus des comparaisons entre les différents modèles d'embedding.

# Embedding

En changeant le modèle d'embedding, les résultats se sont améliorés (voir les différents graphiques comparant les modèles utilisés), mais le modèle peine tout de même à utiliser les bons chunks pour générer sa réponse, et s'emmêle encore, notamment à cause des prompts. Une étude plus poussée des prompts s'est donc avéré pour obtenir des résultats satisfaisants.

Un second aspect avec beaucoup d'importance est la langue. Les modèles d'embedding sont bien meilleurs lorsque la requête est en anglais, où ils arrivent plus facilement à trouver les bons fichiers (même si le fichier lui-même est dans une autre langue). Deux solutions s'offrent alors:
- utiliser un modèle multilangue pouvant gérer le français
- traduire automatiquement les requêtes des utilisateurs

Au final, le modèle d'embedding instructor-large est bien meilleur que même les modèles pourtant en théorie mieux adapté au français.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Sources

## Code 

https://github.com/ollama/ollama/issues/3938 (langchain)

https://github.com/AllAboutAI-YT/easy-local-rag (méthode tierce)

https://github.com/vndee/local-rag-example (langchain)

https://github.com/ollama/ollama/blob/main/docs/tutorials/langchainpy.md (langchain)

https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.similarity_search 
(type de similarity_search)

## Articles

https://iamajithkumar.medium.com/working-with-faiss-for-similarity-search-59b197690f6c

https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7

https://medium.com/@varsha.rainer/document-loaders-in-langchain-7c2db9851123

https://blog.gopenai.com/creating-rag-app-with-llama2-and-chainlit-a-step-by-step-guide-d98499c2cd89

https://blog.gopenai.com/building-a-rag-chatbot-using-llamaindex-groq-with-llama3-chainlit-b1709f770f55

https://aws.amazon.com/fr/what-is/retrieval-augmented-generation/

https://hackernoon.com/fr/un-tutoriel-sur-la-fa%C3%A7on-de-cr%C3%A9er-votre-propre-chiffon-et-de-l%27ex%C3%A9cuter-localement-langchain-ollama-streamlit

## Sources concernant l'embedding

### Code 

https://github.com/PrithivirajDamodaran/FlashRank

https://github.com/voyage-ai/voyage-large-2-instruct/tree/main

https://stackoverflow.com/questions/46849733/change-metadata-of-pdf-file-with-pypdf2 (pour corriger les metadata des fichiers pdf)

https://www.reddit.com/r/LangChain/comments/1ba77pu/difference_between_as_retriever_and_similarity/

### Articles 

https://towardsdatascience.com/openai-vs-open-source-multilingual-embedding-models-e5ccb7c90f05

https://medium.com/@fabio.matricardi/metadata-metamorphosis-from-plain-data-to-enhanced-insights-with-retrieval-augmented-generation-8d1a8d5a6061 (metadata)

https://docs.llamaindex.ai/en/stable/examples/prompts/prompts_rag/ (pour gérer des prompts avec llama_index)

https://www.pinecone.io/learn/openai-embeddings-v3/

https://datacorner.fr/spacy/ (ranking)

https://www.pinecone.io/learn/chunking-strategies/ (chunking techniques)

https://huggingface.co/spaces/mteb/leaderboard (comparaison modèles embedding)

https://atlas.nomic.ai/map/nomic-text-embed-v1-5m-sample

https://platform.openai.com/docs/guides/embeddings

https://www.reddit.com/r/LangChain/comments/186sgyf/rag_filtering_docs_to_only_send_relevant_data_to/

https://blog.devgenius.io/automated-translation-of-text-and-data-in-python-with-deep-translator-d980afee70ab (traduction)

