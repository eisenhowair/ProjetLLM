#!/bin/bash

sudo apt update && sudo apt install pip

pip install -r requirements.txt

curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3:instruct