## Procédure de lancement

- lancer curl -fsSL https://ollama.com/install.sh | sh pour installer ollama
- puis ollama run llama3:8b pour installer le modèle
- http://localhost:11434/ permet de vérifier le lancement du server


   model = Ollama(   
        base_url='http://localhost:11434',
        model="llama3:8b")
        permet d'utiliser l'API REST de Ollama installé localement


## Sources

https://github.com/ollama/ollama/blob/main/docs/tutorials/langchainpy.md

https://github.com/sudarshan-koirala/langchain-ollama-chainlit/blob/main/simple_chaiui.py