FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY worker.py .
COPY requirements.in .
COPY ./nanovllm ./nanovllm

RUN pip install --no-cache-dir -r requirements.in

ENTRYPOINT ["hfendpoint"]
CMD ["python", "worker.py"]
