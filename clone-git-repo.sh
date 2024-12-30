sudo apt install figlet

#printf 'Clone git repository' | figlet -w 200
#git clone git@github.com:sagarsGitArena/shop-talk.git



curl -s https://raw.githubusercontent.com/sagarsGitArena/shop-talk/main/clone-git-repo.sh | bash

if [ -d “shop-talk ]; then
    echo "Directory exists"
	 cd short-talk
	 sh set-up-env.sh
else
    echo "Directory does not exist"
fi


