#!/bin/bash

script="$0"
FOLDER="$(dirname $script)"

source $FOLDER/utils.sh
PROJECT_ROOT="$(abspath $FOLDER/..)"

echo "building base image in folder $PROJECT_ROOT"

docker build -f $PROJECT_ROOT/base_image/Dockerfile \
             -t dominicbreuker/resnet_50_docker_base:latest \
             $PROJECT_ROOT

docker push dominicbreuker/resnet_50_docker_base
