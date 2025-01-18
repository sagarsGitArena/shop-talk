

mkdir -p ./dags
cp -r ./airflow-dag/* ./dags
mkdir -p ./logs ./plugins ./config 
echo -e "AIRFLOW_UID=$(id -u)" > .env
