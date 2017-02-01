#!/bin/bash

script="$0"
FOLDER="$(dirname $script)"

source $FOLDER/utils.sh
PROJECT_ROOT="$(abspath $FOLDER/..)"

BASE_IMAGE_FOLDER=$PROJECT_ROOT/base_image

echo "building base image in folder $BASE_IMAGE_FOLDER"

docker build -f $BASE_IMAGE_FOLDER/Dockerfile \
             -t dominicbreuker/resnet_50_docker_base:latest \
             $BASE_IMAGE_FOLDER

#docker push dominicbreuker/resnet_50_docker_base
