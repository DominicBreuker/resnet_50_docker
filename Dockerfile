FROM dominicbreuker/resnet_50_docker_base:latest

COPY resnet_50/extractor.py /resnet_50/extractor.py
COPY resnet_50/model_test.py /resnet_50/model_test.py
COPY resnet_50/test_images /resnet_50/test_images
COPY resnet_50/imagenet_utils.py /resnet_50/imagenet_utils.py

CMD ["python", "/resnet_50/model_test.py"]
