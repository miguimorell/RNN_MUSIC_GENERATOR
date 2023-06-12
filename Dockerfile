#FROM python:3.10-slim
FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

# COPY requirements.txt requirements.txt
COPY . .
RUN pip install .
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# COPY Makefile Makefile
COPY Makefile Makefile
RUN apt-get update && apt-get install -y make

CMD uvicorn RNN_MUSIC_GENERATOR.Generator.melody_generator:app --host 0.0.0.0 --port $PORT
