#!/bin/bash
# Deploy the CT pipeline cloud runner to Google Cloud Run.
#
# Usage: ./deploy.sh PROJECT_ID BUCKET_PREFIX
#   PROJECT_ID     your Google Cloud project id (created in the console first)
#   BUCKET_PREFIX  prefix for the two buckets, e.g. "ctpipe" ->
#                  gs://ctpipe-uploads and gs://ctpipe-results
#
# This script does NOT create the project or the billing account: those need
# your clicks in the Cloud Console (see cloud/README.md). It fails with a
# plain-English message if they are missing.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: ./deploy.sh PROJECT_ID BUCKET_PREFIX"
  echo "Example: ./deploy.sh my-ct-project ctpipe"
  exit 1
fi
PROJECT_ID="$1"
PREFIX="$2"
REGION="us-central1"
SERVICE="ctpipe-cloud"
UPLOADS_BUCKET="${PREFIX}-uploads"
RESULTS_BUCKET="${PREFIX}-results"

fail() { echo "ERROR: $1" >&2; exit 1; }

# --- preconditions ---------------------------------------------------------
command -v gcloud >/dev/null 2>&1 || fail "gcloud is not installed. Install it from https://cloud.google.com/sdk/docs/install and run 'gcloud auth login'."
gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q . \
  || fail "No active gcloud login. Run 'gcloud auth login' first."
gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1 \
  || fail "Project '$PROJECT_ID' not found or not accessible. Create it at https://console.cloud.google.com/projectcreate first."

BILLING_ACCT="$(gcloud beta billing projects describe "$PROJECT_ID" --format="value(billingAccountName)" 2>/dev/null || true)"
[ -n "$BILLING_ACCT" ] || fail "No billing account linked to '$PROJECT_ID'. In the console: Billing -> Link a billing account. A card is required for verification; you are not charged while inside the free tier."

gcloud config set project "$PROJECT_ID" >/dev/null

# --- APIs ------------------------------------------------------------------
echo "Enabling APIs (run, storage, cloudbuild, artifactregistry)..."
gcloud services enable \
  run.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com

# --- buckets ---------------------------------------------------------------
for B in "$UPLOADS_BUCKET" "$RESULTS_BUCKET"; do
  if gcloud storage buckets describe "gs://$B" >/dev/null 2>&1; then
    echo "Bucket gs://$B already exists."
  else
    echo "Creating bucket gs://$B in $REGION..."
    gcloud storage buckets create "gs://$B" --location="$REGION" --uniform-bucket-level-access
  fi
done

# --- signed URLs need the runtime SA to sign blobs --------------------------
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
echo "Granting $RUNTIME_SA permission to sign URLs..."
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --member="serviceAccount:$RUNTIME_SA" \
  --role="roles/iam.serviceAccountTokenCreator" >/dev/null

# --- build & deploy ----------------------------------------------------------
echo "Building container with Cloud Build..."
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/ctpipe/$SERVICE:latest" \
  --dockerfile cloud/Dockerfile .

echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE" \
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/ctpipe/$SERVICE:latest" \
  --region "$REGION" \
  --platform managed \
  --memory 8Gi --cpu 4 \
  --timeout 900 \
  --concurrency 1 \
  --min-instances 0 --max-instances 2 \
  --allow-unauthenticated \
  --set-env-vars "CLOUD_UPLOADS_BUCKET=$UPLOADS_BUCKET,CLOUD_RESULTS_BUCKET=$RESULTS_BUCKET,CLOUD_ALLOWED_ORIGINS=https://covid-ct-web.vercel.app"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format="value(status.url)")"
echo "Service is live at: $URL"

# --- $1 budget alert -----------------------------------------------------------
BUDGET_NAME="ctpipe-cloud-1usd-cap"
if gcloud billing budgets list --billing-account="$BILLING_ACCT" --filter="displayName=$BUDGET_NAME" --format="value(name)" 2>/dev/null | grep -q .; then
  echo "Budget alert '$BUDGET_NAME' already exists."
else
  echo "Creating \$1 budget alert..."
  gcloud billing budgets create \
    --billing-account="$BILLING_ACCT" \
    --display-name="$BUDGET_NAME" \
    --budget-amount=1USD \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=90 \
    --threshold-rule=percent=100 >/dev/null \
  && echo "Budget alert created: emails you at 50/90/100% of \$1." \
  || echo "WARNING: could not create the budget alert; set one manually in the console (Billing -> Budgets)."
fi

echo
echo "Done. Next:"
echo "  1. Copy the service URL above into web/site/cloud.js as CLOUD_API and push the site."
echo "  2. Upload a DICOM zip on the demo page to run your first cloud analysis."
