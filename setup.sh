#!/bin/bash

#pip install -r requirements.txt

cat .env

current_path=$(pwd)

path_array=(${current_path//\// })

third_directory=${path_array[2]}

new_command="/home/UHA/${third_directory}/.local/bin/huggingface-cli login"

$new_command
