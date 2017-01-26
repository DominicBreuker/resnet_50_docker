FROM dominicbreuker/resnet_50_docker_base:latest

RUN pip install 'tqdm==4.11.2'

COPY resnet_50/extractor.py /resnet_50/extractor.py
COPY resnet_50/result_saver /resnet_50/result_saver

CMD ["python", "/resnet_50/extractor.py", "--help"]
