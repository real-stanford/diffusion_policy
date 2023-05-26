#!/bin/sh
xhost +

if [ ! -z $1 ]; then
  TAG_NAME=$1
else
  TAG_NAME="latest"
fi

export WANDB_API_KEY=`cat wandb_key.txt`

if [ ! -z ${DATASET_DIR} ]; then
  echo "DATASET_DIR is set to ${DATASET_DIR}"
else
  export DATASET_DIR=${PWD}/../data
  echo "DATASET_DIR is not set, using default value instead: ${DATASET_DIR}"
fi

docker-compose up ${TAG_NAME} &
