FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# App code
WORKDIR /app
COPY . /app

# Default: run gateway
EXPOSE 8000
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
