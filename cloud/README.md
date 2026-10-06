# CT pipeline cloud runner

Upload a zip of DICOM files, get the full analysis back: the real pipeline
(`pipeline/`: ingest, validate, eligibility, quantify) running on Google Cloud
Run, with overlay images and a versioned `results.json`.

## How it works

```
browser                    Cloud Run (this service)              Cloud Storage
  |  GET /upload-url            |                                     |
  |---------------------------->|                                     |
  |  signed PUT url + job id    |                                     |
  |<----------------------------|                                     |
  |  PUT dicom.zip ----------------------------------------------->|
  |  POST /jobs {gcs_uri}       |                                     |
  |---------------------------->| download zip, run pipeline          |
  |                             |---------- results.json, PNGs ------>|
  |<----------------------------| {job_id, results, overlay urls}     |
```

- `GET /upload-url?filename=scan.zip` hands out a 15-minute signed upload URL.
  The file goes straight to Cloud Storage, so the 32 MB Cloud Run request
  limit never gets in the way.
- `POST /jobs` runs the pipeline synchronously (a 255-slice series takes
  about 4 minutes) and returns the results plus short-lived signed URLs for
  the overlay images.
- `GET /jobs/{job_id}` re-reads the stored results later.
- Auth is Application Default Credentials only: on Cloud Run that is the
  runtime service account. No keys, no service-account JSON anywhere.

## Cost: the free tier covers it

Per full 255-slice run (4 vCPU, 8 GiB, ~4 minutes):

- ~960 vCPU-seconds and ~1,920 GiB-seconds.
- Always-free monthly allowance: 180,000 vCPU-seconds, 360,000 GiB-seconds,
  2M requests. That is roughly **180 full runs a month** before spending a
  cent, and the service scales to zero when idle so it costs nothing
  between runs.

The deploy script also sets a **$1 budget alert** (emails at 50/90/100%),
so a surprise bill is not possible to miss.

## Deploy

You do five things in the console; the script does the rest.

1. Go to [console.cloud.google.com/projectcreate](https://console.cloud.google.com/projectcreate)
   and create a project. Note the **Project ID**.
2. Billing: link a billing account to the project. A card is required for
   verification; you are not charged while inside the free tier.
3. APIs: open [API Library](https://console.cloud.google.com/apis/library),
   search for each of these and click **Enable**: `Cloud Run`,
   `Cloud Storage`, `Cloud Build`, `Artifact Registry`.
4. On your machine: install the
   [gcloud CLI](https://cloud.google.com/sdk/docs/install) and run
   `gcloud auth login`.
5. From the repo root, run:

```bash
./cloud/deploy.sh YOUR_PROJECT_ID YOUR_BUCKET_PREFIX
```

That builds the container with Cloud Build, deploys to Cloud Run in
`us-central1` (`--memory 8Gi --cpu 4 --timeout 900 --max-instances 2
--min-instances 0`), creates `gs://<prefix>-uploads` and
`gs://<prefix>-results`, grants the runtime service account permission to
sign URLs, and creates the $1 budget alert. It prints the service URL at
the end.

Then paste that URL into `web/site/cloud.js` as `CLOUD_API`, push the site,
and the "Analyze your own scan" section goes live.

## Local test

```bash
docker build -f cloud/Dockerfile -t ctpipe-cloud .
docker run --rm -p 8080:8080 \
  -e CLOUD_STORAGE_BACKEND=local -e CLOUD_LOCAL_DIR=/tmp/cloud-local \
  ctpipe-cloud
# in another terminal:
curl localhost:8080/healthz
```

With the local backend, `gs://` URIs map to `/tmp/cloud-local/`. POST a job
with a zip made from the test fixtures (see `tests/conftest.py`
`write_synthetic_slice`).

## Limits (by design)

- Zip cap: 600 MB compressed, 500 MB unzipped, 1,000 files, 200 MB per file.
- The zip must contain at least one file with a DICOM magic header.
- Zip-slip paths (`..`, absolute paths) are rejected.
- Each pipeline stage gets 10 minutes; the Cloud Run request timeout is
  15 minutes.
- At most 2 concurrent jobs per instance, 2 instances max.
