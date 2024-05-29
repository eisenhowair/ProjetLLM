import requests
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

"""
pip install flask
sudo snap install ngrok
ngrok config add-authtoken (avec le token ici) # pour se connecter (il faut se créer un compte pour en obtenir un)
sudo ngrok http 5000
dans un autre terminal, lancer ce programme

"""
app = Flask(__name__)
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

REPO_NAME = os.getenv("REPO_NAME")
REPO_OWNER = os.getenv("REPO_OWNER")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


llm_local = Ollama(base_url="http://localhost:11434", model="llama3:instruct")


def get_diff(pull_number):
    print("dans get_diff")

    url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pull_number}/files'
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    response = requests.get(url, headers=headers)
    return response.json()


def get_pull_requests():
    print("dans get_pull_requests")

    url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls'
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    response = requests.get(url, headers=headers)
    return response.json()


def generate_comment(diff_files):
    print("dans generate_comment")

    # Simulation de l'appel au modèle Ollama
    # Ici, on suppose que Ollama a une fonction `analyze_diff` qui prend les différences et génère un commentaire

    system_instructions = """
    Vous êtes un assistant développeur français. Évaluez les changements dans le code en termes de qualité, de bonnes pratiques, et de performance. Fournissez une analyse détaillée et des recommandations spécifiques.
    """

    prompt_template = PromptTemplate(
        template="{instructions}\n\Changements dans le code par fichier: {context}",
        input_variables=["instructions", "context"],
    )

    changes = ""
    print(diff_files)
    for file in diff_files:
        filename = file['filename']
        if filename is None:
            filename = ("(pas de nom de fichier)")
        patch = file.get('patch', '(pas de différence détectée)')
        changes += f"Fichier: {filename}\nDiff:\n{patch}\n\n"

    full_prompt = prompt_template.format(
        instructions=system_instructions, context=changes)

    response = llm_local.generate([full_prompt])
    return response


def post_comment(pull_number, comment):
    print("dans post_comment")
    url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues/{pull_number}/comments'
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Content-Type': 'application/json'
    }
    data = {'body': comment}
    response = requests.post(url, headers=headers, json=data)
    return response.json()


@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    if data['action'] in ['opened', 'synchronize']:
        print("PR detected")
        pull_number = data['pull_request']['number']
        diff_files = get_diff(pull_number)
        comment = generate_comment(diff_files)
        response = post_comment(pull_number, comment.generations[0][0].text)
        print(response)
        return jsonify(response)
    return '', 200


if __name__ == '__main__':
    app.run(port=5000)
