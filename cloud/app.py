"""Cloud runner for the DICOM ingestion pipeline.

A visitor uploads a zip of DICOM files; this service runs the real pipeline
(ingest -> validate -> quantify from pipeline/) on Cloud Run and returns the
analysis plus overlay images.

No keys live here. On Cloud Run the service uses Application Default
Credentials (the runtime service account) for all Google Cloud Storage calls.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.log import get_logger, setup_logging  # noqa: E402

setup_logging(os.environ.get("CTPIPE_LOG_LEVEL", "INFO"), "json")
logger = get_logger("cloud.app")

APP_VERSION = "1.0"

# Upload safety caps.
MAX_ZIP_BYTES = 600 * 1024 * 1024  # compressed upload cap
MAX_UNZIPPED_BYTES = 500 * 1024 * 1024
MAX_FILES = 1000
MAX_FILE_BYTES = 200 * 1024 * 1024
STAGE_TIMEOUT_S = 600  # per pipeline stage


class CloudSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLOUD_", extra="ignore")

    storage_backend: str = Field(default="gcs", description="gcs|local")
    uploads_bucket: str = Field(default="", description="GCS bucket for uploads")
    results_bucket: str = Field(default="", description="GCS bucket for results")
    local_dir: str = Field(default="/tmp/cloud-local", description="local backend root")
    allowed_origins: str = Field(
        default="https://covid-ct-web.vercel.app",
        description="comma-separated CORS origins",
    )
    url_expiry_minutes: int = Field(default=15)
    app_root: str = Field(default="/app", description="repo root for pipeline subprocesses")


settings = CloudSettings()

app = FastAPI(title="CT pipeline cloud runner", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# At most two jobs at once per instance; Cloud Run runs max 2 instances.
job_semaphore = asyncio.Semaphore(2)


# ---------------------------------------------------------------- storage ---
class Storage:
    def upload_url(self, gcs_uri: str, expires_min: int) -> str:
        raise NotImplementedError

    def download(self, gcs_uri: str, dest_path: str) -> None:
        raise NotImplementedError

    def upload_file(self, src: str, gcs_uri: str, content_type: str | None = None) -> None:
        raise NotImplementedError

    def read_json(self, gcs_uri: str) -> dict | None:
        raise NotImplementedError

    def write_json(self, gcs_uri: str, payload: dict) -> None:
        raise NotImplementedError

    def signed_get_url(self, gcs_uri: str, expires_min: int) -> str:
        raise NotImplementedError


def _split_uri(gcs_uri: str) -> tuple[str, str]:
    m = re.fullmatch(r"gs://([^/]+)/(.+)", gcs_uri or "")
    if not m:
        raise ValueError(f"not a gs:// URI: {gcs_uri!r}")
    return m.group(1), m.group(2)


class GcsStorage(Storage):
    """Production backend. Auth is ADC only: on Cloud Run that is the runtime
    service account. Signed URLs use the IAM signBlob API, so the runtime
    service account needs roles/iam.serviceAccountTokenCreator on itself
    (deploy.sh grants it)."""

    def __init__(self) -> None:
        from google.cloud import storage

        self._client = storage.Client()

    def _blob(self, gcs_uri: str):
        bucket, name = _split_uri(gcs_uri)
        return self._client.bucket(bucket).blob(name)

    def _sign(self, blob, method: str, expires_min: int) -> str:
        import google.auth
        from google.auth.transport.requests import Request as AuthRequest

        creds, _ = google.auth.default()
        creds.refresh(AuthRequest())
        email = getattr(creds, "service_account_email", None)
        if not email:
            # Local user credentials: fall back to the plain path, which works
            # when the credential carries a private key.
            return blob.generate_signed_url(
                version="v4", expiration=timedelta(minutes=expires_min), method=method
            )
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expires_min),
            method=method,
            service_account_email=email,
            access_token=creds.token,
        )

    def upload_url(self, gcs_uri: str, expires_min: int) -> str:
        return self._sign(self._blob(gcs_uri), "PUT", expires_min)

    def download(self, gcs_uri: str, dest_path: str) -> None:
        self._blob(gcs_uri).download_to_filename(dest_path)

    def upload_file(self, src: str, gcs_uri: str, content_type: str | None = None) -> None:
        blob = self._blob(gcs_uri)
        if content_type:
            blob.content_type = content_type
        blob.upload_from_filename(src)

    def read_json(self, gcs_uri: str) -> dict | None:
        from google.cloud.exceptions import NotFound

        try:
            return json.loads(self._blob(gcs_uri).download_as_bytes())
        except NotFound:
            return None

    def write_json(self, gcs_uri: str, payload: dict) -> None:
        blob = self._blob(gcs_uri)
        blob.content_type = "application/json"
        blob.upload_from_string(json.dumps(payload, indent=2))

    def signed_get_url(self, gcs_uri: str, expires_min: int) -> str:
        return self._sign(self._blob(gcs_uri), "GET", expires_min)


class LocalStorage(Storage):
    """Test backend: maps gs://bucket/name to <local_dir>/bucket/name."""

    def __init__(self, root: str) -> None:
        self._root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, gcs_uri: str) -> str:
        bucket, name = _split_uri(gcs_uri)
        if ".." in name.split("/"):
            raise ValueError("bad object name")
        p = os.path.join(self._root, bucket, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def upload_url(self, gcs_uri: str, expires_min: int) -> str:
        return "local://" + gcs_uri

    def download(self, gcs_uri: str, dest_path: str) -> None:
        src = self._path(gcs_uri)
        if not os.path.isfile(src):
            raise FileNotFoundError(gcs_uri)
        shutil.copyfile(src, dest_path)

    def upload_file(self, src: str, gcs_uri: str, content_type: str | None = None) -> None:
        shutil.copyfile(src, self._path(gcs_uri))

    def read_json(self, gcs_uri: str) -> dict | None:
        p = self._path(gcs_uri)
        if not os.path.isfile(p):
            return None
        with open(p) as fh:
            return json.load(fh)

    def write_json(self, gcs_uri: str, payload: dict) -> None:
        with open(self._path(gcs_uri), "w") as fh:
            json.dump(payload, fh, indent=2)

    def signed_get_url(self, gcs_uri: str, expires_min: int) -> str:
        return "local://" + gcs_uri


def get_storage() -> Storage:
    if settings.storage_backend == "local":
        return LocalStorage(settings.local_dir)
    if not settings.uploads_bucket or not settings.results_bucket:
        raise RuntimeError("CLOUD_UPLOADS_BUCKET and CLOUD_RESULTS_BUCKET must be set")
    return GcsStorage()


# ------------------------------------------------------------------ models ---
class JobRequest(BaseModel):
    gcs_uri: str = Field(description="gs://<uploads-bucket>/uploads/<job_id>/<file>.zip")


class JobError(Exception):
    """A job failed for a reason we can explain in plain language."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# --------------------------------------------------------------- pipeline ---
def _run_stage(name: str, args: list[str], workdir: str) -> None:
    """Run one pipeline CLI stage; raise JobError with its stderr tail."""
    logger.info("stage_start", stage=name)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", f"pipeline.{name}"] + args,
            cwd=settings.app_root,
            capture_output=True,
            text=True,
            timeout=STAGE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise JobError(f"The {name} step took too long (over 10 minutes).")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        logger.error("stage_failed", stage=name, returncode=proc.returncode, tail=tail)
        raise JobError(f"The {name} step failed: {tail.strip().splitlines()[-1] if tail.strip() else 'unknown error'}")
    logger.info("stage_done", stage=name)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=settings.app_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _has_dicom_magic(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(132)
        return len(head) >= 132 and head[128:132] == b"DICM"
    except OSError:
        return False


def validate_and_extract(zip_path: str, dest_dir: str) -> dict:
    """Zip-slip-safe extraction with caps; returns file inventory."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise JobError("That file is not a valid zip archive.")
    infos = zf.infolist()
    if len(infos) > MAX_FILES:
        raise JobError(f"Too many files in the zip ({len(infos)}); the cap is {MAX_FILES}.")
    total = 0
    safe: list[zipfile.ZipInfo] = []
    for info in infos:
        if info.is_dir():
            continue
        name = info.filename
        if name.startswith("/") or ".." in name.split("/"):
            raise JobError("Unsafe file path inside the zip.")
        total += info.file_size
        if info.file_size > MAX_FILE_BYTES:
            raise JobError(f"One file is over the 200 MB per-file cap: {name}.")
        safe.append(info)
    if total > MAX_UNZIPPED_BYTES:
        raise JobError(
            f"Unzipped size {total / 1e6:.0f} MB is over the 500 MB cap."
        )
    if not safe:
        raise JobError("The zip is empty.")
    os.makedirs(dest_dir, exist_ok=True)
    for info in safe:
        target = os.path.join(dest_dir, info.filename)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    zf.close()

    # At least one file must look like DICOM (magic "DICM" at offset 128).
    checked = 0
    for info in safe[:50]:
        p = os.path.join(dest_dir, info.filename)
        if os.path.isfile(p):
            checked += 1
            if _has_dicom_magic(p):
                break
    else:
        raise JobError(
            "No DICOM files found in the zip. Upload a zip of CT DICOM slices "
            "(files starting with the DICOM magic header)."
        )
    return {"n_files": len(safe), "unzip_bytes": total}


def make_overlays(src_path: str, out_dir: str, n_images: int = 5) -> list[str]:
    """Render lung-window CT + infection overlay PNGs for representative slices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, settings.app_root)
    from pipeline.quantify import load_volume, segment_infection, segment_lungs

    vol = load_volume(src_path)
    lung = segment_lungs(vol)
    inf = segment_infection(vol)
    lung_per_slice = lung.reshape(lung.shape[0], -1).sum(axis=1)
    idx = [i for i, c in enumerate(lung_per_slice) if c > 0]
    if not idx:
        return []
    picks = sorted({idx[int(len(idx) * q)] for q in (0.1, 0.3, 0.5, 0.7, 0.9)})[:n_images]

    os.makedirs(out_dir, exist_ok=True)
    names = []
    for k, i in enumerate(picks):
        hu = vol[i].astype(np.float32)
        # Lung window: WL -600, WW 1500 -> [-1350, 150].
        img = np.clip((hu + 1350) / 1500.0, 0, 1)
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        mask = (inf[i] > 0) & (lung[i] > 0)
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = (1, 0, 0, 0.55)
        ax.imshow(rgba)
        ax.axis("off")
        ax.set_title(f"slice {i}", fontsize=10, color="#666")
        name = f"overlay_{k:02d}_slice_{i:03d}.png"
        fig.savefig(os.path.join(out_dir, name), bbox_inches="tight", facecolor="black")
        plt.close(fig)
        names.append(name)
    return names


def run_job(job_id: str, gcs_uri: str, store: Storage) -> dict:
    """Download, validate, run the pipeline, upload results. Returns results.json."""
    tmp = tempfile.mkdtemp(prefix=f"cloudjob_{job_id}_")
    zip_path = os.path.join(tmp, "input.zip")
    dicom_dir = os.path.join(tmp, "dicom")
    work_dir = os.path.join(tmp, "work")
    os.makedirs(work_dir, exist_ok=True)
    try:
        logger.info("job_download", job_id=job_id, gcs_uri=gcs_uri)
        try:
            store.download(gcs_uri, zip_path)
        except Exception as exc:
            raise JobError(f"Could not download the upload: {exc}", status_code=404)
        if os.path.getsize(zip_path) > MAX_ZIP_BYTES:
            raise JobError("The uploaded zip is over the 600 MB cap.")

        inv = validate_and_extract(zip_path, dicom_dir)

        _run_stage("ingest", ["--dicom-root", dicom_dir, "--out-dir", work_dir], tmp)
        manifest = os.path.join(work_dir, "manifest.parquet")
        _run_stage("validate", ["--manifest", manifest, "--out-dir", work_dir], tmp)
        with open(os.path.join(work_dir, "validation_report.json")) as fh:
            vreport = json.load(fh)
        _run_stage(
            "quantify",
            ["--manifest", manifest, "--out-dir", work_dir,
             "--results-name", "quantification.parquet"],
            tmp,
        )

        import pandas as pd

        rdf = pd.read_parquet(os.path.join(work_dir, "quantification.parquet"))
        from pipeline import contracts
        from pipeline.quantify import code_hash

        overlay_files: list[str] = []
        series_out = []
        for _, row in rdf.iterrows():
            src = str(row.get("source") or "")
            ov_dir = os.path.join(tmp, "overlays")
            ovs = make_overlays(src, ov_dir) if src and os.path.exists(src) else []
            overlay_files.extend(ovs)
            series_out.append(
                {
                    "series_uid": str(row["series_uid"]),
                    "slices": int(row["slices"]),
                    "lung_voxels": int(row["lung_voxels"]),
                    "infected_voxels": int(row["infected_voxels"]),
                    "infection_pct": round(float(row["infection_pct"]), 2),
                    "ci_low": round(float(row["ci_low"]), 2),
                    "ci_high": round(float(row["ci_high"]), 2),
                    "overlays": ovs,
                }
            )
            for name in ovs:
                store.upload_file(
                    os.path.join(ov_dir, name),
                    f"gs://{settings.results_bucket}/results/{job_id}/{name}",
                    content_type="image/png",
                )

        results = {
            "job_id": job_id,
            "status": "done",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pipeline": {
                "code_hash": code_hash(),
                "git_sha": _git_sha(),
                "results_contract": contracts.RESULTS_CONTRACT,
            },
            "input": {
                "gcs_uri": gcs_uri,
                "n_files": inv["n_files"],
                "n_series": vreport["n_series"],
            },
            "validation": {
                "n_series": vreport["n_series"],
                "n_eligible": vreport["n_eligible"],
                "n_quarantined": vreport["n_quarantined"],
                "failures": vreport["validation_failures"],
                "quarantined": vreport["quarantined"],
            },
            "series": series_out,
        }
        prefix = f"gs://{settings.results_bucket}/results/{job_id}"
        store.write_json(f"{prefix}/results.json", results)
        store.write_json(
            f"{prefix}/status.json",
            {"job_id": job_id, "status": "done",
             "at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        )
        logger.info("job_done", job_id=job_id, n_series=len(series_out))
        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _with_overlay_urls(results: dict, store: Storage) -> dict:
    out = dict(results)
    urls = {}
    for s in results.get("series", []):
        for name in s.get("overlays", []):
            gcs = f"gs://{settings.results_bucket}/results/{results['job_id']}/{name}"
            urls[name] = store.signed_get_url(gcs, settings.url_expiry_minutes)
    out["overlay_urls"] = urls
    return out


# ------------------------------------------------------------------- api ---
@app.get("/healthz")
def healthz():
    return {"ok": True, "version": APP_VERSION}


@app.get("/upload-url")
def upload_url(filename: str):
    """Signed PUT URL for a DICOM zip. Returns the URL, the job id, and the
    gs:// URI the frontend must pass to POST /jobs after uploading."""
    base = os.path.basename(filename or "")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:80] or "scan.zip"
    if not safe.lower().endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file of DICOM slices.")
    job_id = uuid.uuid4().hex[:16]
    store = get_storage()
    gcs_uri = f"gs://{settings.uploads_bucket}/uploads/{job_id}/{safe}"
    try:
        url = store.upload_url(gcs_uri, settings.url_expiry_minutes)
    except Exception as exc:
        logger.exception("upload_url_failed")
        raise HTTPException(500, f"Could not make an upload URL: {exc}")
    return {"upload_url": url, "gcs_uri": gcs_uri, "job_id": job_id,
            "expires_minutes": settings.url_expiry_minutes}


@app.post("/jobs")
async def create_job(req: JobRequest):
    store = get_storage()
    prefix = f"gs://{settings.uploads_bucket}/uploads/"
    if not req.gcs_uri.startswith(prefix) or not req.gcs_uri.endswith(".zip"):
        raise HTTPException(400, "gcs_uri must be an uploads/ zip from /upload-url.")
    job_id = req.gcs_uri[len(prefix):].split("/")[0]
    if not re.fullmatch(r"[0-9a-f]{16}", job_id or ""):
        raise HTTPException(400, "Could not parse a job id from gcs_uri.")
    status_uri = f"gs://{settings.results_bucket}/results/{job_id}/status.json"
    store.write_json(status_uri, {
        "job_id": job_id, "status": "running",
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    try:
        async with job_semaphore:
            results = await asyncio.to_thread(run_job, job_id, req.gcs_uri, store)
    except JobError as exc:
        store.write_json(status_uri, {"job_id": job_id, "status": "failed",
                                      "error": exc.message})
        raise HTTPException(exc.status_code, exc.message)
    except Exception:
        logger.exception("job_crashed", job_id=job_id)
        store.write_json(status_uri, {"job_id": job_id, "status": "failed",
                                      "error": "The analysis crashed unexpectedly."})
        raise HTTPException(500, "The analysis crashed unexpectedly.")
    return _with_overlay_urls(results, store)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{16}", job_id or ""):
        raise HTTPException(400, "Bad job id.")
    store = get_storage()
    prefix = f"gs://{settings.results_bucket}/results/{job_id}"
    results = store.read_json(f"{prefix}/results.json")
    if results:
        return _with_overlay_urls(results, store)
    status = store.read_json(f"{prefix}/status.json")
    if status:
        return status
    raise HTTPException(404, "No such job.")
