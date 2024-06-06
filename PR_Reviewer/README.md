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
Dans cette section, tu pourras expliquer ce qu'est le script **PR_Reviewer**. Parle de son but, de son utilité et du contexte dans lequel il est utilisé.

**Exemple :**
Le script **PR_Reviewer** est conçu pour automatiser la revue des Pull Requests (PR) sur GitHub. Il analyse les changements proposés, identifie les erreurs potentielles et fournit des recommandations pour améliorer le code. Cela facilite le processus de revue de code en réduisant le temps et l'effort nécessaires.

### 2. Difficultés rencontrées
Ici, tu pourras décrire les défis et les problèmes que tu as rencontrés lors de la création de **PR_Reviewer**. Parle des aspects techniques ou des obstacles que tu as dû surmonter.

**Exemple :**
Lors de la création de **PR_Reviewer**, plusieurs défis ont été rencontrés :
- Gestion des cas particuliers dans les PR complexes.
- Intégration avec l'API de GitHub pour récupérer et poster des commentaires.
- Optimisation des performances pour traiter de grandes quantités de données en temps réel.

### 3. Comment utiliser le programme
Cette section doit fournir des instructions claires sur la façon d'utiliser **PR_Reviewer**. Inclut des exemples de commandes, des paramètres, et des étapes d'installation.

**Exemple :**
Pour utiliser **PR_Reviewer**, suivez les étapes suivantes :

1. Clonez le dépôt :
   ```sh
   git clone https://github.com/votrecompte/PR_Reviewer.git
