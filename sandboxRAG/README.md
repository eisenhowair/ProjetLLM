## dossier_entier_RAG.py

Question par console fonctionne bien en utilisant un index, mais ne fonctionne pas avec chainlit car n'utilise pas l'index.

## RAGPdfOrTxt.py

Ajoute un fichier à la fois à l'index, fonctionne bien avec chainlit, mais lent.

# Dossier tierces

- differents_textes comportent plusieurs textes de longueur et type différent, pour vérifier le bon fonctionnement des indexs.
- data contient l'index. Le supprimer n'est pas un problème, car les programmes en recrééront un.

## Sources

### Code 

https://github.com/ollama/ollama/issues/3938

https://github.com/vndee/local-rag-example

https://github.com/AllAboutAI-YT/easy-local-rag

https://github.com/vndee/local-rag-example

### Articles

https://iamajithkumar.medium.com/working-with-faiss-for-similarity-search-59b197690f6c

https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7

https://medium.com/@varsha.rainer/document-loaders-in-langchain-7c2db9851123

https://blog.gopenai.com/creating-rag-app-with-llama2-and-chainlit-a-step-by-step-guide-d98499c2cd89

https://aws.amazon.com/fr/what-is/retrieval-augmented-generation/

https://hackernoon.com/fr/un-tutoriel-sur-la-fa%C3%A7on-de-cr%C3%A9er-votre-propre-chiffon-et-de-l%27ex%C3%A9cuter-localement-langchain-ollama-streamlit