#!/bin/bash

script="$0"
FOLDER="$(pwd)/$(dirname $script)"

source $FOLDER/utils.sh
PROJECT_ROOT="$(abspath $FOLDER/..)"
echo "project root folder $PROJECT_ROOT"

echo "build docker image"
/bin/bash $FOLDER/build.sh

##### RUN #####
echo "Starting container..."

docker run --rm \
           -it \
           dominicbreuker/resnet_50_docker:latest \
           /bin/sh -c "python /resnet_50/model/model_test.py"
           #/bin/bash
