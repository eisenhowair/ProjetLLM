import os

def find_project_root(start_path):
    current_path = start_path
    while current_path != os.path.abspath(os.sep):
        if '.git' in os.listdir(current_path):
            return current_path
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        if parent_path == current_path:
            break
        current_path = parent_path
    raise FileNotFoundError("Dossier .git non trouvé dans l'arborescence.")
try:
    start_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(start_path)
    print(f"Le chemin du dossier ProjetLLM est : {project_root}")
    path_vectorstores = project_root+"/vectorstores/"
    print(path_vectorstores)
except Exception as e:
    print(f"Erreur : {str(e)}")


embedding_model_hf_en_mpnet = "sentence-transformers/all-mpnet-base-v2"
index_en_path_mpnet = path_vectorstores+"MPNet_vectorstore"

embedding_model_hf_fr = "dangvantuan/sentence-camembert-base"
index_fr_path_camembert =path_vectorstores+"Camembert_vectorstore"

embedding_model_hf_en_instructor_large = "hkunlp/instructor-large"
index_en_path_instructor_large = path_vectorstores+"Instructor_large_vectorstore"

embedding_model_hf_en_instructor_base = "hkunlp/instructor-base"
index_en_path_instructor_base = path_vectorstores+"Instructor_base_vectorstore"

embedding_model_hf_en_instructor_xl = "hkunlp/instructor-xl"
index_en_path_instructor_xl =path_vectorstores+"Instructor_xl_vectorstore"

embedding_model_ol_en_nomic = "nomic-embed-text"
index_en_path_OL = path_vectorstores+"OL_vectorstore"  # nomic-embed
