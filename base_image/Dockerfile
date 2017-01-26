FROM tensorflow/tensorflow:0.12.1

RUN pip install -U pip && \
    pip install 'keras==1.2.1' && \
    pip install 'h5py==2.6.0' && \
    pip install 'pillow==4.0.0'

COPY resnet_50/model /resnet_50/model/

RUN ["python", "/resnet_50/model/download_weights.py"]

RUN ["python", "/resnet_50/model/model_test.py"]
