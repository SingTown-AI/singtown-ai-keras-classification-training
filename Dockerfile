from nvcr.io/nvidia/tensorflow:25.01-tf2-py3

RUN pip install keras singtown-ai

COPY . .

RUN python cache_weight.py
