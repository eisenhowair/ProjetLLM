## RAG scraping

Chatbot construit avec chainlit qui réponds à vos question sur un site web en lui donnant son URL.

### Technologies

- **Scraping** :
	- BeautifulSoup

- **RAG** :
	- ChromaDB 
	- ParentDocumentRetriever 
	- RecursiveCharacterTextSplitter

- **Modèles** :
	- llama3:instruct
	- hkunlp/instructor-large

- **Interface** :
	- chainlit

### Pour commencer
 1. `pip install -r requirements.txt`
 2. `chainlit run main.py`