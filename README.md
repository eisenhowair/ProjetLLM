<h3 align="center">Projet "Personnalisation et déploiement de modèles de langage"</h3>
<p align="center">Le but de ce projet est de découvrir les différentes manières de personnaliser un modèles de langage.</p>
<br/>


<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>Sommaire</strong></summary>
  <ol>
    <li>
      <a href="#à-propos-du-projet">À propos du projet</a>
      <ul>
        <li><a href="#technologies">Technologies</a></li>
      </ul>
    </li>
    <li>
      <a href="#pour-commencer">Pour commencer</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#utilisation">Utilisation</a></li>
    <li><a href="#contacts">Contacts</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## À propos du projet

Dans ce projet on explore trois types de personnalisation de modèle de langage. 

La personnalisation par prompt spécifique qui permet de diriger le modèle simplement en lui donnant des instructions spécifiques sous forme de prompt. 

Le Retrivial Augmented Generation (RAG) permet de personnaliser un modèle en lui donnant une source de données supplémentaires sans le ré-entraîner. Les données supplémentaires sont divisées en blocs puis sont vectorisées et stockées pour permettre au modèle de les utiliser. 

Le finetuning est le troisième type de personnalisation. Cette méthode permet d'affiner l'intelligence d'un modèle pour le rendre plus efficace sur une tâche spécifique.

Nous avons réalisé plusieurs applications en utilisant ces trois types de personnalisation.


### Technologies
[![Static Badge](https://img.shields.io/badge/Langchain-blue?style=for-the-badge&link=https%3A%2F%2Fwww.langchain.com%2F)](https://www.langchain.com/)

[![Static Badge](https://img.shields.io/badge/LLAMAINDEX-lightblue?style=for-the-badge)
](https://www.llamaindex.ai/)

[![Static Badge](https://img.shields.io/badge/Chainlit-pink?style=for-the-badge&link=https%3A%2F%2Fwww.langchain.com%2F)
](https://chainlit.io/)

[![Static Badge](https://img.shields.io/badge/PYTORCH-orange?style=for-the-badge)
](https://pytorch.org/)

[![Static Badge](https://img.shields.io/badge/FAISS-white?style=for-the-badge)
](https://ai.meta.com/tools/faiss/)

[![Static Badge](https://img.shields.io/badge/CHROMADB-lightgrey?style=for-the-badge)
](https://www.trychroma.com/)


<!-- GETTING STARTED -->
## Pour commencer

### Installation

Lancez le script  shell `setup.sh` pour installer automatiquement  tous les éléments nécessaires pour ce projet.

OU

Suivez ces étapes pour mettre en place le projet.
Depuis la racine du projet :

1.  Installez les modules python :
	- `pip install -r requirements.txt`
2. Installez Ollama et récupérer le modèle
	- `curl -fsSL https://ollama.com/install.sh | sh`  (source : [Ollama](https://ollama.com/download))
	- `ollama pull llama3:instruct`

Certaines applications ont leur propre **requirements.txt** qui contient des modules supplémentaires.
Enfin, chaque dossier disposant d'un fichier **.env.example** nécessite un fichier **.env**, dont le contenu sera indiqué dans le **.env.example**, et dans le **README.md** associé.


<!-- USAGE EXAMPLES -->
## Utilisation

Chaque dossier représente une ou plusieurs applications. Vous trouverez un README dans chaque dossier pour l'utilisation et les exigences supplémentaires.

| Application  | Dossier | Type(s) de personnalisation |
|-----------|-----------|-----------|
| Chatbot simple | [/simpleChat](https://github.com/eisenhowair/ProjetLLM/tree/main/simpleChat) | Prompts spécifiques |
| Chatbot assistant e-mail | [/mailAssistantByPrompts](https://github.com/eisenhowair/ProjetLLM/tree/main/mailAssistantByPrompts) | Prompts spécifiques |
| Extension web zimbra | [/ollama-reply-zimbra](https://github.com/eisenhowair/ProjetLLM/tree/main/ollama-reply-zimbra) | Prompts spécifiques |
| Chatbot génération d'exercice de mathématiques | [/genereExoMath](https://github.com/eisenhowair/ProjetLLM/tree/main/genereExoMath) | Prompts spécifiques + RAG |
| PR Reviewer  | [/PR_Reviewer](https://github.com/eisenhowair/ProjetLLM/tree/main/PR_Reviewer) | Prompts spécifiques + RAG |
| Chatbot RAG ENT, pdf | [/sandboxRAG](https://github.com/eisenhowair/ProjetLLM/tree/main/sandboxRAG) | Prompts spécifiques + RAG |
| Chatbot RAG pdf, url | [/sandboxRAG2](https://github.com/eisenhowair/ProjetLLM/tree/main/sandboxRAG2) | Prompts spécifiques + RAG |
| Finetuning phi2 | [/sandboxFinetuning](https://github.com/eisenhowair/ProjetLLM/tree/main/sandboxFinetuning) | Prompts spécifiques + Finetuning |


## Remarques

Tous les programmes bénéficiant d'une interface de connexion ont "elias" comme nom d'utilisateur et mot de passe. Ces valeurs sont changeables dans la fonction `auth_callback` du programme lancé.

Attention car si un programme utilisant un index a un problème pendant la création dudit index, le dossier contenant l'index aura été crée, mais vide. De ce fait, le programme, lors de sa prochaine exécution, va penser que l'index existe car le dossier existe, ce qui n'est pas le cas. Il renverra donc une erreur de type:
```bash
ValueError: No existing llama_index.vector_stores.faiss.base found at vectorstores/llama_index_mpnet/default__vector_store.json.
```
Pour éviter cette erreur, il est généralement plus simple de supprimer le dossier vide.


<!-- CONTACT -->
## Contacts

Elias Mouaheb - elias.mouaheb@uha.fr

Théo Nicod - theo.nicod@uha.fr
