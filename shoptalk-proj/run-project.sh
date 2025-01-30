WORK_DIR=`pwd`
echo $WORK_DIR
docker-compose --env-file $WORK_DIR/.env up --build

