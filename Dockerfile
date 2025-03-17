from tensorflow/tensorflow:2.19.0-gpu

RUN pip install keras singtown-ai==0.4.3

COPY . .

RUN python cache_weight.py
