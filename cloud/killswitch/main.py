"""Billing kill-switch: disables billing when the $1 budget hits 100%.

Triggered by the Pub/Sub topic that the billing budget publishes to.
The budget notification schema (1.0) carries alertThresholdExceeded as a
1.0-based fraction; 1.0 means spend reached 100% of the budget.

Disabling billing stops all billable services on the project (Cloud Run
goes dark) until billing is re-linked by hand. That is the point: it makes
further spend very hard. Budget alerts and the kill-switch can each be
delayed by a few minutes, so this is a strong safety net rather than a
guaranteed hard cap; the uploads on/off switch is the primary protection.

Re-enable with:
    gcloud beta billing projects link PROJECT_ID \
        --billing-account=BILLING_ACCOUNT_ID
or in the console: Billing -> Link a billing account.
"""

import base64
import json
import os

import functions_framework
from googleapiclient import discovery

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")


@functions_framework.cloud_event
def kill_billing(cloud_event):
    raw = cloud_event.data["message"]["data"]
    data = json.loads(base64.b64decode(raw).decode("utf-8"))
    threshold = float(data.get("alertThresholdExceeded") or 0)
    cost = data.get("costAmount")
    budget = data.get("budgetAmount")
    print(
        f"budget notification: threshold={threshold} cost={cost} budget={budget}",
        flush=True,
    )
    if threshold < 1.0:
        print("below the 100% threshold; no action taken", flush=True)
        return "ok"
    if not PROJECT_ID:
        print("GCP_PROJECT_ID not set; refusing to act", flush=True)
        return "misconfigured"
    billing = discovery.build("billing", "v1", cache_discovery=False)
    billing.projects().updateBillingInfo(
        name=f"projects/{PROJECT_ID}", body={"billingAccountName": ""}
    ).execute()
    print(f"billing disabled for project {PROJECT_ID}", flush=True)
    return "billing-disabled"
