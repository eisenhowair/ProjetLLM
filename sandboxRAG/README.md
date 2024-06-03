# RAG

## RAG_moodle_ENT.py

Anciennement : Question par console fonctionne bien en utilisant un index, mais ne fonctionne pas avec chainlit car n'utilise pas l'index.
Version actuelle (23 mai): Le programme fonctionne maintenant avec chainlit, et ne met que 3 minutes à répondre aux questions, en se basant sur un index composé de plusieurs fichiers de taille et type variable. Le modèle d'embedding n'est plus nomic-embed-text, qui faisait trop d'erreur dans sa sélection de documents, mais aussi de chunks, mais instructor-large, qui est bien plus consistant. L'index est stocké localement pour ne pas avoir à le recréer à chaque exécution du programme.

A été ajouté un système de connexion permettant de consulter les anciennes conversations, ainsi que de suivre l'exécution des requêtes via literal.ai (en ayant un fichier .env correspondant aux attentes demandées par le README dans modele_personnalise)

## RAGPdfOrTxt.py

Ajoute un fichier à la fois à l'index, fonctionne bien avec chainlit, mais lent.

# Dossier tierces

- differents_textes comportent plusieurs textes de longueur et type différent, pour vérifier le bon fonctionnement des indexs.
- data contient l'index. Le supprimer n'est pas un problème, car les programmes en recrééront un.

## Sources

### Code 

https://github.com/ollama/ollama/issues/3938 (langchain)

https://github.com/AllAboutAI-YT/easy-local-rag (méthode tierce)

https://github.com/vndee/local-rag-example (langchain)

https://github.com/ollama/ollama/blob/main/docs/tutorials/langchainpy.md (langchain)

https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.similarity_search 
(type de similarity_search)

### Articles

https://iamajithkumar.medium.com/working-with-faiss-for-similarity-search-59b197690f6c

https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7

https://medium.com/@varsha.rainer/document-loaders-in-langchain-7c2db9851123

https://blog.gopenai.com/creating-rag-app-with-llama2-and-chainlit-a-step-by-step-guide-d98499c2cd89

https://aws.amazon.com/fr/what-is/retrieval-augmented-generation/

https://hackernoon.com/fr/un-tutoriel-sur-la-fa%C3%A7on-de-cr%C3%A9er-votre-propre-chiffon-et-de-l%27ex%C3%A9cuter-localement-langchain-ollama-streamlit

# Embedding

En changeant le modèle d'embedding, les résultats se sont améliorés(voir les différents graphiques comparant les modèles utilisés), mais le modèle peine otut de même à utiliser les bons chunks pour générer sa réponse, et s'emmêle encore, notamment à cause des prompts. Une étude plus poussée des prompts s'avère donc nécessaire.

Un second aspect avec beaucoup d'importance est la langue. Les modèles d'embedding sont bien meilleurs lorsque la requête est en anglais, où ils arrivent plus facilement à trouver les bons fichiers (même si le fichier lui-même est dans une autre langue). Deux solutions s'offrent alors:
- utiliser un modèle multilangue pouvant gérer le français
- traduire automatiquement les requêtes des utilisateurs

## Sources

### Code 

https://github.com/PrithivirajDamodaran/FlashRank

https://github.com/voyage-ai/voyage-large-2-instruct/tree/main

https://stackoverflow.com/questions/46849733/change-metadata-of-pdf-file-with-pypdf2 (pour corriger les metadata des fichiers pdf)

https://www.reddit.com/r/LangChain/comments/1ba77pu/difference_between_as_retriever_and_similarity/

### Articles 

https://towardsdatascience.com/openai-vs-open-source-multilingual-embedding-models-e5ccb7c90f05

https://www.pinecone.io/learn/openai-embeddings-v3/

https://datacorner.fr/spacy/ (ranking)

https://www.pinecone.io/learn/chunking-strategies/ (chunking techniques)

https://huggingface.co/spaces/mteb/leaderboard (comparaison modèles embedding)

https://atlas.nomic.ai/map/nomic-text-embed-v1-5m-sample

https://platform.openai.com/docs/guides/embeddings

https://www.reddit.com/r/LangChain/comments/186sgyf/rag_filtering_docs_to_only_send_relevant_data_to/

https://blog.devgenius.io/automated-translation-of-text-and-data-in-python-with-deep-translator-d980afee70ab (traduction)

