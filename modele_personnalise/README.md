<!-- GETTING STARTED -->
<a name="readme-top"></a>

## Introduction

Le but des programmes de ce dossier est de mettre en place un générateur d'exercices de mathématiques, qui permet ensuite d'aider à la résolution 
de l'exercice, en donnant des indices, ou en corrigeant l'exercice après un certain nombre de tentatives.

Trois versions du programme existent, mais exo_math_3prompts.py sert de version la plus aboutie, avec _prep_message_chainlit_3prompts.py_ pour répartir le code.
Les 2 autres fichiers de code représentent des tentatives, chacun avec ses défauts. 

### Installation

1. Avant de pouvoir lancer le programme, il est nécessaire d'installer les dépendances et logiciels requis, trouvables dans le fichier requirements.txt du dossier parent

```sh
  pip install -r ../requirements.txt
  ```

2. Il faudra ensuite télécharger le modèle correspondant

```sh
  ollama pull llama3:instruct
  ```

3. Enfin, se créer un fichier .env avec les variables LITERAL_API_KEY et CHAINLIT_AUTH_SECRET, avec la première nécessitant de se créer un compte sur LiteralAI (puis aller dans Settings -> General Default Key), et la seconde qui est trouvable en tapant 
 ```sh
  chainlit create-secret
  ```


## Fonctionnement

### Initialisation du programme

Au lancement du programme sont initialisées 2 variables servant de mémoire, l'une globale, qui sera utile pour quitter la discussion et y revenir, et une plus courte,
qui retient les messages d'une discussion portant sur un exercice, et se vide dès lors que l'exercice change, pour que le modèle ait accès à un contexte au moment de répondre. Ces 2 variables sont enrichies à chaque parution de message, qu'il soit de l'utilisateur ou de l'IA. Un message demandant les loisirs de l'utilisateur apparait,
et la réponse de l'utilisateur est enregistrée, car elle sera utilisée pour générer les exercices.

### Génération d'exercice

La fonction setup_exercice_model() commence par récupérer la discussion actuelle, puis les loisirs de l'utilisateur. Elle va utiliser ses données, ainsi qu'un prompt spécifiquement écrit pour que le modèle soit à même de créer des exercices suivant certaines règles, et va les passer à un objet Runnable à renvoyer.

Le Runnable renvoyé est utilisé pour récupérer une réponse du modèle, qui sera donc un exercice de mathématique. L'exercice est enregistré dans la session, pour que le correcteur y ait accès facilement. Enfin le flag "compris" lui aussi dans la session est passé à False, pour appeler le correcteur.

### Résolution d'exercice

La fonction setup_aide_model() récupère elle aussi la discussion en mémoire, puis renvoit un Runnable avec un prompt plus précis que lors de la génération, car c'est à partir de ce prompt que le modèle communiquera avec l'utilisateur la plupart du temps. Le prompt est censé aider l'utilisateur par le biais d'indices, sans toutefois donner la réponse, mais au bout d'un certain nombre de tentatives (3 ici), le modèle donne la réponse.

Une fois le message d'aide envoyé, la fonction verifie_comprenhension est appelée, demandant à l'utilisateur s'il a compris la réponse. Si oui, "compris" passe à True, et on rappelle le générateur, sinon, le modèle doit continuer de fournir des indices ou la réponse, suivant le nombre de tentatives.

En repassant au générateur, la mémoire courte est vidée.

### Autres

Le programme dispose aussi de fonctions permettant de quitter la discussion pour la reprendre plus tard, ou de changer l'âge de l'utilisateur aux yeux du modèle. Il est à noter que bien que l'IA ait conscience de l'âge, elle n'arrive pas à générés des exercices qui y sont pertinents.


## Intérêt par rapport aux précédentes versions

Le fichier exo_math_3prompts.p utilise un prompt pour aier l'utilisateur avec les indices, et un autre pour donner la réponse. Cela créait des malentendus dans certains cas, en plus de ne pas être très fluide.

<p align="right">(<a href="#readme-top">back to top</a>)</p>