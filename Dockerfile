from tensorflow/tensorflow:2.19.0-gpu

RUN pip install keras singtown-ai==0.5.0

COPY cache_weight.py cache_weight.py
COPY train.py train.py

RUN python cache_weight.py
