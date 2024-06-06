# ollama-reply-zimbra

Tout d'abord, cette extension est tirée de [ollama-reply](https://github.com/jose-donato/ollama-reply/blob/main). Elle a simplement été adapté pour zimbra (zimbra par défaut, sans personnalisation).


### Prérequis :

voir [ici](https://github.com/jose-donato/ollama-reply/blob/main/README.md#prerequisites)

### Installation

Pour l'installation, vous pouvez suivre les instructions depuis le [readme ollama-reply](https://github.com/jose-donato/ollama-reply/tree/main?tab=readme-ov-file#installation-steps) ou le guide d'installation ci-dessous qui prend en compte les problèmes que nous avons rencontrés :

1. Installez [Ollama](https://ollama.com/download)
2. Par défaut, l'application Ollama est lancée. On ne souhaite pas lancer Ollama via l'application.`sudo systemctl stop ollama` pour stopper l'application. 
3. Ajoutez la variable d'environnement OLLAMA_ORIGINS avec `export OLLAMA_ORIGINS=*`.
4. Dans le même terminal, lancez localement ollama sans l'application avec `ollama serve`
5. Installez le modèle llama3:8b via un autre terminal avec `ollama pull llama3:8b`
6. Dans votre navigateur (basé sur chromium), ouvrez la page des extensions (chrome://extensions/).  Activez le mode développeur en haut à droite. Cliquez sur 'Charger l'extension non empaquetée' et sélectionnez le dossier ollama-reply-zimbra/dist de ce dépôt.

### Utilisation

Une fois que vous êtes sur votre boîte mail zimbra, cliquez sur un mail. Ouvrez l'extension en haut à droite.
La section 'indications' (facultative) permet de donner des indications pour orienter la réponse du modèle.
La section 'ton' permet de définir l'intonation de la réponse du modèle.
Cliquez sur le bouton générer pour obtenir une réponse au mail ouvert actuellement (cela peut parfois prendre une à deux minutes).
Ne fermez pas l'extension avant d'avoir reçu la réponse 
