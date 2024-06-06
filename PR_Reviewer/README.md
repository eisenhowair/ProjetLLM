
https://blog.gopenai.com/building-a-rag-chatbot-using-llamaindex-groq-with-llama3-chainlit-b1709f770f55 (chainlit avec llama index)

https://ogre51.medium.com/context-window-of-language-models-a530ffa49989 (context window)

https://docs.chainlit.io/integrations/llama-index ( documentation chainlit)

https://docs.llamaindex.ai/en/stable/examples/prompts/prompts_rag/ (prompt llama_index)

  

https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/ (doc officielle)

https://github.com/joaomdmoura/crewAI

https://docs.crewai.com/tools/GitHubSearchTool (impossible d'importer crewAI)

https://lightning.ai/lightning-ai/studios/chat-with-your-code-using-rag (a donné des pistes)

  

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
- après que l'écran noir de ngrok soit apparu, une url sera proposée à droite. C'est cette url, à laquelle il faudra rajouter "/webhook", qu'il faudra entrer dans le webhook de Github.

> [!WARNING]
> Attention, l'url de ngrok change quotidiennement pour la version gratuite, il faut donc penser à mettre à jour l'url du webhook sur Github à chaque fois avec la nouvelle url fournie par ngrok.


  
## Github_chatbot

### Introduction
