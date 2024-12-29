# You must build the base image with ../tensorflow.Dockerfile
FROM tensorflow:latest 

RUN pip3 install wandb
COPY train.py /workspace/train.py
COPY configs /workspace/configs
COPY utils /workspace/utils
COPY data /workspace/data
COPY models /workspace/models
