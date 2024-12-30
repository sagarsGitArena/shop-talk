###### NVIDIA - DOCKER installation to leverage host GPUs #########
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && \
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2


#docker pull nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
To verify nvidia docker isntallation and to check the underlying GPUs are leveraged , try running the following command
docker run --rm --gpus all  nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 nvidia-smi
#nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi

docker run --rm --gpus all  nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 nvidia-smi
nvidia-smi
docker info | grep -i runtime
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
