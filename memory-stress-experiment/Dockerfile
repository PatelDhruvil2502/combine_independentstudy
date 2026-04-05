FROM python:3.10-slim
RUN apt-get update && apt-get install -y build-essential python3-dev
RUN pip install hnswlib numpy
COPY worker.py /app/worker.py
COPY real_world_dataset.npy /app/real_world_dataset.npy
WORKDIR /app
CMD ["python", "worker.py"]
