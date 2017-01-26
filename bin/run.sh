#!/bin/bash

script="$0"
FOLDER="$(pwd)/$(dirname $script)"

source $FOLDER/utils.sh
PROJECT_ROOT="$(abspath $FOLDER/..)"
echo "project root folder $PROJECT_ROOT"

echo "build docker image"
/bin/bash $FOLDER/build.sh

##### VOLUMES #####

# folder containing data
DATA_DIR=$PROJECT_ROOT/data
echo "Using data in $DATA_DIR"

##### RUN #####
echo "Starting container..."

docker run --rm \
           -it \
           -p 8888:8888 \
           -v $DATA_DIR:/data \
           dominicbreuker/resnet_50_docker:latest
