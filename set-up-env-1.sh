
#ssh-keygen -t ed25519 -C "vasamsetty@gmail.com"
git clone git@github.com:sagarsGitArena/shop-talk.git
sudo apt update
sudo apt upgrade -y
#sudo apt install python3-pip
#pip install boto3
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
docker --version
sudo usermod -aG docker $USER
docker  run hello-world
id -nG
#exit
#ssh to the ec2 again
#docker run hello-world
#id -nG
sudo apt install docker-compose