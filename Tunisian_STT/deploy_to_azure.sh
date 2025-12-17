#!/bin/bash

# Azure Deployment Script for STT Tunisian
# Part of the WIE_TSYP project (shared infrastructure)

set -e  # Exit on error

echo "=================================================="
echo "Azure Deployment - STT Tunisian"
echo "Part of WIE_TSYP Project"
echo "=================================================="
echo ""

# Configuration
export RESOURCE_GROUP="wie_tsyp"
export ACR_NAME="wietsypregistry"  # Must be globally unique, lowercase, no special chars
export LOCATION="eastus"  # Change if needed
export IMAGE_NAME="stt-tunisian"
export CONTAINER_NAME="stt-container"

echo "Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  ACR Name: $ACR_NAME"
echo "  Location: $LOCATION"
echo "  Image Name: $IMAGE_NAME"
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not found. Installing..."
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
else
    echo "✓ Azure CLI found"
fi

# Check if logged in
echo ""
echo "Checking Azure login status..."
if ! az account show &> /dev/null; then
    echo "Not logged in. Opening browser for authentication..."
    az login
else
    echo "✓ Already logged in to Azure"
    az account show --output table
fi

echo ""
read -p "Continue with deployment? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Create Resource Group (if it doesn't exist)
echo ""
echo "Step 1: Creating Resource Group..."
if az group show --name $RESOURCE_GROUP &> /dev/null; then
    echo "✓ Resource group '$RESOURCE_GROUP' already exists"
else
    echo "Creating resource group '$RESOURCE_GROUP' in $LOCATION..."
    az group create --name $RESOURCE_GROUP --location $LOCATION
    echo "✓ Resource group created"
fi

# Create ACR (if it doesn't exist)
echo ""
echo "Step 2: Creating Azure Container Registry..."
if az acr show --name $ACR_NAME &> /dev/null; then
    echo "✓ ACR '$ACR_NAME' already exists"
else
    echo "Creating ACR '$ACR_NAME'..."
    az acr create \
        --resource-group $RESOURCE_GROUP \
        --name $ACR_NAME \
        --sku Basic \
        --admin-enabled true
    echo "✓ ACR created"
fi

# Verify ACR admin is enabled
echo "Ensuring ACR admin is enabled..."
az acr update -n $ACR_NAME --admin-enabled true > /dev/null
echo "✓ ACR admin enabled"

# Build and push image to ACR
echo ""
echo "Step 3: Building and pushing Docker image to ACR..."
echo "This will take 5-10 minutes (uploading ~1.5GB)..."
az acr build \
    --registry $ACR_NAME \
    --image $IMAGE_NAME:v1 \
    --image $IMAGE_NAME:latest \
    .

echo "✓ Image built and pushed to ACR"

# Verify image
echo ""
echo "Verifying image in ACR..."
az acr repository show-tags --name $ACR_NAME --repository $IMAGE_NAME --output table

# Get ACR credentials
echo ""
echo "Step 4: Getting ACR credentials..."
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)

echo "✓ ACR Login Server: $ACR_LOGIN_SERVER"

# Deploy to Azure Container Instances
echo ""
echo "Step 5: Deploying to Azure Container Instances..."

# Check if container already exists
if az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME &> /dev/null; then
    echo "Container '$CONTAINER_NAME' already exists. Deleting old instance..."
    az container delete \
        --resource-group $RESOURCE_GROUP \
        --name $CONTAINER_NAME \
        --yes
    echo "✓ Old container deleted"
fi

echo "Creating new container instance..."
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $ACR_LOGIN_SERVER/$IMAGE_NAME:latest \
    --os-type Linux \
    --cpu 2 \
    --memory 4 \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --dns-name-label stt-tunisian-api \
    --ports 8000 \
    --environment-variables MODEL_PATH=/app/model/STT_Tun_Model

echo "✓ Container created"

# Get the public URL
echo ""
echo "Step 6: Getting deployment information..."
FQDN=$(az container show \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --query ipAddress.fqdn \
    --output tsv)

echo ""
echo "=================================================="
echo "🎉 Deployment Successful!"
echo "=================================================="
echo ""
echo "Your STT API is available at:"
echo ""
echo "  Base URL:     http://$FQDN:8000"
echo "  Health:       http://$FQDN:8000/health"
echo "  API Docs:     http://$FQDN:8000/docs"
echo "  Transcribe:   POST http://$FQDN:8000/transcribe"
echo ""
echo "=================================================="
echo ""

# Wait for container to be ready
echo "Waiting for container to start (model loading takes ~60 seconds)..."
echo "You can monitor logs with:"
echo "  az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --follow"
echo ""

# Save FQDN to file for easy access
echo $FQDN > .azure_fqdn
echo "✓ FQDN saved to .azure_fqdn"


echo ""
echo "To view logs:"
echo "  az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo ""
echo "To stop the container:"
echo "  az container stop --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo ""
echo "To delete the container:"
echo "  az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes"
echo ""
echo "To check the container status:"
echo "az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query instanceView.state"
echo ""
echo "To start the container:"
echo "az container start --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo ""
echo "To test the container:"
echo "python3 test_api.py http://$FQDN:8000 felfel0.wav"
echo ""

