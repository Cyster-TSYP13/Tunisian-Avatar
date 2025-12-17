FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENABLE_SYNTHESIS=true
# Replace with your actual Hugging Face API key
ENV HUGGINGFACE_API_KEY=hf_xxxx 
ENV HUGGINGFACE_MODEL=google/flan-t5-base
ENV HF_MAX_NEW_TOKENS=256

WORKDIR /app

# Install system deps for some packages (chroma/onnxruntime may need build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

EXPOSE 8080

# Use uvicorn to serve the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
