#!/bin/bash

set -e

IMAGE_NAME="quantum_privacy_engine"
TAG="latest"
DATA_DIR="./data/financial_transactions"
MODEL_DIR="./gan_models"
RESULTS_DIR="./gan_results"

echo "Building Docker image..."
docker build -t $IMAGE_NAME:$TAG .

echo "Starting container..."
docker run -it --rm \
  -v $(pwd)/$DATA_DIR:/app/data \
  -v $(pwd)/$MODEL_DIR:/app/gan_models \
  -v $(pwd)/$RESULTS_DIR:/app/gan_results \
  -p 5000:5000 \
  $IMAGE_NAME:$TAG

echo "Deployment completed successfully."
