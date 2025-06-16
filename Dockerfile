FROM python:3.10

# Set working directory
WORKDIR /app

# Install OS-level dependencies (keras-ocr butuh libgl)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy semua isi proyek ke container
COPY . /app

# Upgrade pip dan install requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan FastAPI app
CMD ["uvicorn", "ML2.ml_main:app", "--host", "0.0.0.0", "--port", "9000"]
