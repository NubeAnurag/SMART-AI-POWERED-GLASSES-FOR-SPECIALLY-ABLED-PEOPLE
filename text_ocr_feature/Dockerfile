FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-hin \
    python3-tk \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install paddlepaddle==2.5.2 paddleocr easyocr pytesseract opencv-python

# Set the working directory
WORKDIR /app

# Copy your code into the container
COPY . /app

# Set environment variable for PaddlePaddle
ENV FLAGS_use_mkldnn=false

# Default command
CMD ["python", "camera_ocr_live_ocr.py"] 