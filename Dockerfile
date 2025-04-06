FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt

COPY . /app

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "source.fast_api_service:app", "--host", "0.0.0.0", "--port", "8080"]

