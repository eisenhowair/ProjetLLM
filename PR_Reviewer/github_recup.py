import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

REPO_NAME = os.getenv("REPO_NAME")
REPO_OWNER = os.getenv("REPO_OWNER")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Initialize the tool for semantic searches within a specific GitHub repository
tool = GithubSearchTool(
    github_repo='https://github.com/{REPO_OWNER}/{REPO_NAME}',
    content_types=['code', 'issue']  # Options: code, repo, pr, issue
)

# OR

# Initialize the tool for semantic searches within a specific GitHub repository, so the agent can search any repository if it learns about during its execution
tool = GithubSearchTool(
    content_types=['code', 'issue']  # Options: code, repo, pr, issue
)

print(tool)
