#!/bin/bash

# Set the project ID and region
PROJECT_ID="amfam-claims"
REGION="us-central1"

# Build the Docker image
gcloud builds submit --tag us-docker.pkg.dev/$PROJECT_ID/container/amfam-claims --source .

# Deploy the image to Cloud Run
gcloud run deploy amfam-claims \
  --image us-docker.pkg.dev/$PROJECT_ID/container/amfam-claims \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated
