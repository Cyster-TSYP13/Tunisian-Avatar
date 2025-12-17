# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by Vosk
RUN apt-get update && apt-get install -y \
  wget \
  unzip \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY testing.py .

# Create model directory
RUN mkdir -p /app/model

# Copy the Vosk model
# Note: Make sure to extract STT_Tun_Model.zip before building
COPY model/STT_Tun_Model /app/model/STT_Tun_Model

# Set environment variables
ENV MODEL_PATH=/app/model/STT_Tun_Model
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
