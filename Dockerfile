FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

CMD ["python", "container_process_backbone.py"]