#!/bin/bash

# Set the project ID and region
PROJECT_ID="YOUR PROJECT ID HERE"
REGION="us-central1"
SERVICE_ACCOUNT="YOUR SA HERE"
APP_NAME="YOUR APP NAME HERE"

# Build the Docker image
gcloud builds submit --tag us-docker.pkg.dev/$PROJECT_ID/$APP_NAME:latest

# Deploy to Cloud Run
gcloud run deploy $APP_NAME \
  --project $PROJECT_ID \
  --image us-docker.pkg.dev/$PROJECT_ID/$APP_NAME:latest \
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
