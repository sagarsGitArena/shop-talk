
./clean-dirs.sh

mkdir -p ./dags
cp -r ./airflow-dag/* ./dags
mkdir -p ./logs ./plugins ./config ./data/raw ./data/rawimages
echo -e "AIRFLOW_UID=$(id -u)" > .env
