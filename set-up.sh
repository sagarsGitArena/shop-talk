#!/bin/bash

# Check if figlet is installed
if ! dpkg -l | grep -q figlet; then
    sudo apt update
    sudo apt install -y figlet
else
    echo "Figlet is already installed."
fi

# Clone the repository if it doesn't already exist
if [ ! -d "/home/ubuntu/test/shop-talk" ]; then
    printf "\e[31m$(figlet -w 200 "Cloning Git Repo")\e[0m\n"
    git clone git@github.com:sagarsGitArena/shop-talk.git
else
    printf "\e[31m$(figlet -w 200 "Git Repo Already Cloned")\e[0m\n"
fi

# List the current directory's content
pwd
ls -lart 

# Check if the directory exists
if [ -d "/home/ubuntu/test/shop-talk" ]; then
    echo "Directory exists"
    cd /home/ubuntu/test/shop-talk || exit 1
    sh set-up-env.sh
else
    echo "Directory does not exist"
fi
