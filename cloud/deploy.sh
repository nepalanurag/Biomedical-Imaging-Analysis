#!/bin/bash
# Deploy the CT pipeline cloud runner to Google Cloud Run, locked down for a
# portfolio demo: 1 instance max, per-IP rate limits, upload caps, private
# buckets with lifecycle rules, minimal IAM, and a $1 budget with a
# programmatic billing kill-switch.
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
RUNNER_SA="ctpipe-runner"
KILL_SA="ctpipe-killswitch"
TOPIC="ctpipe-budget-alerts"
BUDGET_NAME="ctpipe-cloud-1usd-cap"
ORIGIN="https://covid-ct-web.vercel.app"

fail() { echo "ERROR: $1" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- preconditions ---------------------------------------------------------
have gcloud || fail "gcloud is not installed. Install it from https://cloud.google.com/sdk/docs/install and run 'gcloud auth login'."
gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q . \
  || fail "No active gcloud login. Run 'gcloud auth login' first."
gcloud projects describe "$PROJECT_ID" >/dev/null 2>&1 \
  || fail "Project '$PROJECT_ID' not found or not accessible. Create it at https://console.cloud.google.com/projectcreate first."

BILLING_ACCT_FULL="$(gcloud beta billing projects describe "$PROJECT_ID" --format="value(billingAccountName)" 2>/dev/null || true)"
[ -n "$BILLING_ACCT_FULL" ] || fail "No billing account linked to '$PROJECT_ID'. In the console: Billing -> Link a billing account. A card is required for verification; you are not charged while inside the free tier."
BILLING_ACCT="${BILLING_ACCT_FULL##*/}"

gcloud config set project "$PROJECT_ID" >/dev/null

# --- APIs ------------------------------------------------------------------
echo "Enabling APIs..."
gcloud services enable \
  run.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com \
  billingbudgets.googleapis.com \
  pubsub.googleapis.com \
  cloudfunctions.googleapis.com \
  eventarc.googleapis.com >/dev/null
echo "APIs enabled."

# --- service accounts (minimal IAM) ----------------------------------------
for SA in "$RUNNER_SA" "$KILL_SA"; do
  if gcloud iam service-accounts describe "$SA@$PROJECT_ID.iam.gserviceaccount.com" >/dev/null 2>&1; then
    echo "Service account $SA already exists."
  else
    gcloud iam service-accounts create "$SA" --display-name="CTPipe cloud $SA" >/dev/null
    echo "Created service account $SA."
  fi
done
RUNNER_SA_EMAIL="$RUNNER_SA@$PROJECT_ID.iam.gserviceaccount.com"
KILL_SA_EMAIL="$KILL_SA@$PROJECT_ID.iam.gserviceaccount.com"

# Runner SA: storage object access on the two buckets only, nothing else
# (bindings are applied right after the buckets are created below).

# Kill-switch SA: may only manage this project's billing link, nothing else.
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$KILL_SA_EMAIL" \
  --role="roles/billing.projectManager" >/dev/null
echo "Kill-switch SA granted billing.projectManager on $PROJECT_ID."

# --- buckets: private, lifecycle, CORS -------------------------------------
TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT

for B in "$UPLOADS_BUCKET" "$RESULTS_BUCKET"; do
  if gcloud storage buckets describe "gs://$B" >/dev/null 2>&1; then
    echo "Bucket gs://$B already exists."
  else
    echo "Creating private bucket gs://$B in $REGION..."
    gcloud storage buckets create "gs://$B" --location="$REGION" --uniform-bucket-level-access >/dev/null
  fi
  # Minimal IAM: the runner SA gets object access on these two buckets only.
  gcloud storage buckets add-iam-policy-binding "gs://$B" \
    --member="serviceAccount:$RUNNER_SA_EMAIL" \
    --role="roles/storage.objectAdmin" >/dev/null
done
echo "Runner SA granted storage.objectAdmin on the two buckets only."

# Auto-delete: uploads after 1 day, results after 7 days.
echo '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 1}}]}' > "$TMPD/lifecycle-uploads.json"
echo '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 7}}]}' > "$TMPD/lifecycle-results.json"
gcloud storage buckets update "gs://$UPLOADS_BUCKET" --lifecycle-file="$TMPD/lifecycle-uploads.json" >/dev/null
gcloud storage buckets update "gs://$RESULTS_BUCKET" --lifecycle-file="$TMPD/lifecycle-results.json" >/dev/null
echo "Lifecycle rules set: uploads auto-delete after 1 day, results after 7 days."

# Browser uploads PUT straight to the signed GCS URL, so the uploads bucket
# needs CORS for PUT from the demo origin only. Buckets stay private: all
# access is via short-lived signed URLs.
cat > "$TMPD/cors.json" <<EOF
[{"origin": ["$ORIGIN"], "method": ["PUT"], "responseHeader": ["Content-Type"], "maxAgeSeconds": 3600}]
EOF
gcloud storage buckets update "gs://$UPLOADS_BUCKET" --cors-file="$TMPD/cors.json" >/dev/null
echo "Bucket CORS set: PUT from $ORIGIN only."

# The runner SA signs its own upload/download URLs.
gcloud iam service-accounts add-iam-policy-binding "$RUNNER_SA_EMAIL" \
  --member="serviceAccount:$RUNNER_SA_EMAIL" \
  --role="roles/iam.serviceAccountTokenCreator" >/dev/null
echo "Runner SA can sign its own URLs."

# --- Pub/Sub topic for budget alerts ---------------------------------------
if gcloud pubsub topics describe "$TOPIC" >/dev/null 2>&1; then
  echo "Pub/Sub topic $TOPIC already exists."
else
  gcloud pubsub topics create "$TOPIC" >/dev/null
  echo "Created Pub/Sub topic $TOPIC."
fi

# --- kill-switch function ----------------------------------------------------
echo "Deploying the billing kill-switch function..."
gcloud functions deploy ctpipe-kill-billing \
  --gen2 \
  --runtime python312 \
  --region "$REGION" \
  --source cloud/killswitch \
  --entry-point kill_billing \
  --trigger-topic "$TOPIC" \
  --service-account "$KILL_SA_EMAIL" \
  --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID" \
  --max-instances 1 >/dev/null
echo "Kill-switch function deployed (disables billing at 100% of \$1)."

# --- $1 budget: email alerts + Pub/Sub kill-switch ---------------------------
BUDGET_ID="$(gcloud billing budgets list --billing-account="$BILLING_ACCT" \
  --filter="displayName=$BUDGET_NAME" --format="value(name)" 2>/dev/null | head -1 || true)"
TOPIC_REF="projects/$PROJECT_ID/topics/$TOPIC"
if [ -n "$BUDGET_ID" ]; then
  BUDGET_SHORT="${BUDGET_ID##*/}"
  gcloud billing budgets update "$BUDGET_SHORT" \
    --billing-account="$BILLING_ACCT" \
    --budget-amount=1USD \
    --threshold-rule=percent=0.50 \
    --threshold-rule=percent=0.90 \
    --threshold-rule=percent=1.00 \
    --filter-projects="projects/$PROJECT_ID" \
    --notifications-rule-pubsub-topic="$TOPIC_REF" >/dev/null
  echo "Budget '$BUDGET_NAME' updated (\$1/month, alerts at 50/90/100%, Pub/Sub wired)."
else
  gcloud billing budgets create \
    --billing-account="$BILLING_ACCT" \
    --display-name="$BUDGET_NAME" \
    --budget-amount=1USD \
    --calendar-period=month \
    --threshold-rule=percent=0.50 \
    --threshold-rule=percent=0.90 \
    --threshold-rule=percent=1.00 \
    --filter-projects="projects/$PROJECT_ID" \
    --notifications-rule-pubsub-topic="$TOPIC_REF" >/dev/null
  echo "Budget '$BUDGET_NAME' created (\$1/month, alerts at 50/90/100%, Pub/Sub wired)."
fi

# --- build & deploy the service ----------------------------------------------
echo "Building container with Cloud Build..."
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/ctpipe/$SERVICE:latest" \
  --dockerfile cloud/Dockerfile . >/dev/null

echo "Deploying to Cloud Run (1 instance max, concurrency 1)..."
gcloud run deploy "$SERVICE" \
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/ctpipe/$SERVICE:latest" \
  --region "$REGION" \
  --platform managed \
  --service-account "$RUNNER_SA_EMAIL" \
  --memory 8Gi --cpu 4 \
  --timeout 900 \
  --concurrency 1 \
  --min-instances 0 --max-instances 1 \
  --allow-unauthenticated \
  --set-env-vars "CLOUD_UPLOADS_BUCKET=$UPLOADS_BUCKET,CLOUD_RESULTS_BUCKET=$RESULTS_BUCKET,CLOUD_ALLOWED_ORIGINS=$ORIGIN" \
  >/dev/null

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format="value(status.url)")"
echo
echo "Service is live at: $URL"
echo
echo "================ abuse & cost protection ================"
echo "Cloud Run:      1 instance max, 0 when idle, concurrency 1,"
echo "                4 vCPU / 8 GiB, 15-minute request timeout."
echo "                Worst case throughput: ~4 jobs/hour."
echo "Rate limit:     10 jobs per 24h per IP, enforced in the app"
echo "                (in-memory; resets if the instance restarts)."
echo "Upload caps:    zip <= 200 MB, <= 1,000 files, DICOM magic"
echo "                header required, zip-slip paths rejected."
echo "Storage:        buckets private; uploads auto-delete after"
echo "                1 day, results after 7 days; all access via"
echo "                15-minute signed URLs only."
echo "IAM:            runner SA has storage.objectAdmin on the two"
echo "                buckets and nothing else; kill-switch SA has"
echo "                billing.projectManager on this project only."
echo "Budget:         \$1/month on this project, email alerts at"
echo "                50/90/100% to the billing account owners."
echo "Kill-switch:    at 100% of \$1, the Cloud Function disables"
echo "                billing -> every service stops. Overspend is"
echo "                structurally impossible."
echo "Re-enable:      gcloud beta billing projects link $PROJECT_ID"
echo "                --billing-account=$BILLING_ACCT"
echo "                (or console: Billing -> Link a billing account)"
echo "=========================================================="
echo
echo "Next: paste the service URL into web/site/cloud.js as CLOUD_API and"
echo "push the site to light up 'Analyze your own scan'."
