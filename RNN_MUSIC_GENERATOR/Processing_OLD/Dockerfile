#FROM python:3.10-slim
FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

# COPY requirements.txt requirements.txt

COPY . .
RUN pip install .
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements_prod.txt

# COPY Makefile Makefile
COPY Makefile Makefile
RUN apt-get update && apt-get install -y make
RUN make reset_local_files

CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
