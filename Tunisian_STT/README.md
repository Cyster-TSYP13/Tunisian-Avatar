# 🇹🇳 Tunisian STT API Service

This repository contains a production-ready **FastAPI** service for **Tunisian Arabic Speech-to-Text (ASR)**. It wraps a specialized Vosk/Kaldi model (`STT_Tun_Model`) into a REST API and provides complete configuration for deployment on **Azure**.

## 🚀 Features

*   **FastAPI Backend**: High-performance, asynchronous REST API.
*   **Tunisian Dialect Support**: Optimized for Tunisian Arabic with code-switching (French/English).
*   **Dockerized**: Complete `Dockerfile` for consistent environments.
*   **Azure Ready**: Scripts and documentation for deploying to Azure Container Instances (ACI).
*   **Streaming Support**: Real-time transcription capabilities.

## 📂 Project Structure

```
Tunisian_STT/
├── app.py                # FastAPI application
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
├── deploy_to_azure.sh    # Automated Azure deployment script
├── testing.py            # CLI script for testing the model
└── test_api.py           # Script to test the deployed API
```

## 🛠️ Local Setup

### Prerequisites

*   Python 3.10+
*   Docker (optional, for containerization)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Model

You can download the model directly using `curl`:

```bash
sudo apt-get install curl

curl -L https://huggingface.co/Sali7a8603/Tunisian_STT/resolve/main/STT_Tun_Model.zip \
     --output STT_Tun_Model.zip
```

Then extract it and organize the directory structure:

```bash
unzip STT_Tun_Model.zip
mkdir -p model
mv STT_Tun_Model model/
```

### 3. Run Locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Access the API documentation at `http://localhost:8000/docs`.

---

## ☁️ Azure Deployment

This project includes a fully automated script to deploy the service to **Azure Container Instances (ACI)**.

### Prerequisites

*   **Azure CLI**: [Install Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
*   **Azure Account**: You need an active subscription.

### Deployment Steps

1.  **Login to Azure**:
    ```bash
    az login
    ```

2.  **Run the Deployment Script**:
    The `deploy_to_azure.sh` script handles everything: creating the resource group, container registry (ACR), building the image, and deploying the container.

    ```bash
    chmod +x deploy_to_azure.sh
    ./deploy_to_azure.sh
    ```

    **What the script does:**
    *   Creates a Resource Group (`wie_tsyp`).
    *   Creates an Azure Container Registry (`wietsypregistry`).
    *   Builds the Docker image (`stt-tunisian`) and pushes it to ACR.
    *   Deploys the container to Azure Container Instances with 2 vCPUs and 4GB RAM.
    *   Sets the `MODEL_PATH` environment variable to `/app/model/STT_Tun_Model`.

3.  **Access Your API**:
    Once deployed, the script will output your API URL (FQDN).
    
    Example: `http://stt-tunisian-api.eastus.azurecontainer.io:8000`

### Manual Deployment (Optional)

If you prefer to run commands manually, here is the workflow:

```bash
# 1. Variables
RG="my-group"
ACR="my-registry"
IMAGE="stt-tunisian"

# 2. Build Image
az acr build --registry $ACR --image $IMAGE:latest .

# 3. Deploy Container
az container create \
    --resource-group $RG \
    --name stt-container \
    --image $ACR.azurecr.io/$IMAGE:latest \
    --dns-name-label stt-tunisian-api \
    --ports 8000 \
    --cpu 2 --memory 4 \
    --environment-variables MODEL_PATH=/app/model/STT_Tun_Model
```

## 🧪 Testing

### Test the Model (CLI)

```bash
python testing.py path/to/model path/to/audio.wav
```

### Test the API

```bash
python test_api.py http://<your-api-url>:8000 path/to/audio.wav
```

## 📜 License

Apache 2.0
