
figlet 'update'
sudo apt update
figlet 'upgrade'
sudo apt upgrade -y

figlet 'Install Docker'
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
sudo rm -f /usr/share/keyrings/docker-archive-keyring.gpg
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null 
sudo apt update
printf "Install the following : \n\t 1> Docker Comuity,\n\t 2> Docker CLI,\n\t 3> Container Runtime" | figlet -w 100
sudo apt install -y docker-ce docker-ce-cli containerd.io

printf "Verify Docker version" | figlet -w 200
docker --version
sudo usermod -aG docker $USER

printf "Verify Docker is functional" | figlet -w 200
docker  run hello-world
id -nG
#exit
#ssh to the ec2 again
#docker run hello-world
#id -nG

printf "Install docker-compose" | figlet -w 150
sudo apt install docker-compose
