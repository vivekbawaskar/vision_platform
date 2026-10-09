# Container for hosted demos (Hugging Face Spaces, Render, Railway, ...).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    DB_BACKEND=auto \
    DEFAULT_SOURCE="Upload Video" \
    SQLITE_PATH=/tmp/vision_platform.db \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image small (the default wheel bundles CUDA).
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Bake the model weights into the image so the first start is fast.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

EXPOSE 7860
CMD ["sh", "-c", "streamlit run main.py --server.port=${PORT} --server.address=0.0.0.0"]
