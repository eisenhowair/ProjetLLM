#!/bin/bash

# pour pc de la fac

sudo apt install pip

pip install -r requirements.txt

cat .env

current_path=$(pwd)

path_array=(${current_path//\// })

third_directory=${path_array[2]}

new_command="/home/UHA/${third_directory}/.local/lib/python3.8/site-packages/huggingface-cli login"

$new_command

curl -fsSL https://ollama.com/install.sh | sh
