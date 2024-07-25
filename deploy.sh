#!/bin/bash

# Set the project ID and region
PROJECT_ID="amfam-claims"
REGION="us-central1"
SERVICE_ACCOUNT="function@amfam-claims.iam.gserviceaccount.com"
APP_NAME="claims"

# Build the Docker image
gcloud builds submit --tag us-docker.pkg.dev/$PROJECT_ID/container/amfam-claims --source .

gcloud run deploy $APP_NAME \
  --project $PROJECT_ID \
  --image us-docker.pkg.dev/$PROJECT_ID/container/$APP_NAME:latest \
  --platform managed \
  --region $REGION \
  --port 8080 \
  --cpu 1 \
  --memory 1000Mi \
  --concurrency 80 \
  --timeout 3000 \
  --service-account $SERVICE_ACCOUNT \
  --allow-unauthenticated \
  --clear-env-vars
