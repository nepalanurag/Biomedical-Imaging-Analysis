"""Central configuration for the pipeline (pydantic-settings).

Every setting has a sensible default and can be overridden, in increasing
order of precedence:

1. defaults below,
2. a ``.env`` file in the repo root,
3. environment variables prefixed with ``CTPIPE_`` (e.g. ``CTPIPE_OUT_DIR``),
4. explicit CLI flags (each CLI only applies flags the user actually passed).

Secrets (API keys) come from the environment only, never from CLI flags,
so they do not leak into shell history or process listings.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CTPIPE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths ---
    dicom_root: str = Field(default="", description="DICOM tree to ingest")
    out_dir: str = Field(default="pipeline_data", description="pipeline output dir")
    manifest_name: str = Field(default="manifest.parquet")
    results_dir: str = Field(default="pipeline_data/results")
    write_nifti: bool = Field(default=True)

    # --- segmentation bands (HU) ---
    lung_lo: int = Field(default=-950)
    lung_hi: int = Field(default=-300)
    inf_lo: int = Field(default=-700)
    inf_hi: int = Field(default=-200)

    # --- sweep ---
    sweep_out_dir: str = Field(default="pipeline_data/sweep")

    # --- second reader ---
    second_reader_out_dir: str = Field(default="pipeline_data/second_reader")
    vlm_provider: str = Field(default="gemini", description="gemini|openai|dryrun")
    vlm_model: str = Field(default="gemini-2.0-flash")
    google_api_key: str = Field(
        default="", description="env GOOGLE_API_KEY or CTPIPE_GOOGLE_API_KEY"
    )
    openai_api_key: str = Field(
        default="", description="env OPENAI_API_KEY or CTPIPE_OPENAI_API_KEY"
    )

    # --- logging ---
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json", description="json|text")

    @field_validator("lung_lo", "lung_hi", "inf_lo", "inf_hi")
    @classmethod
    def _hu_range(cls, v: int) -> int:
        if not -1100 <= v <= 3200:
            raise ValueError(f"HU band edge {v} outside plausible range")
        return v

    @field_validator("log_level")
    @classmethod
    def _level(cls, v: str) -> str:
        v = v.upper()
        if v not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ValueError(f"unknown log level {v}")
        return v

    def validate_bands(self) -> None:
        if not self.lung_lo < self.lung_hi:
            raise ValueError(f"lung band inverted: [{self.lung_lo},{self.lung_hi}]")
        if not self.inf_lo < self.inf_hi:
            raise ValueError(f"infection band inverted: [{self.inf_lo},{self.inf_hi}]")


def apply_cli_overrides(settings: PipelineSettings, args, fields) -> PipelineSettings:
    """Copy explicitly-passed argparse values onto the settings object.

    Argparse defaults must be None for this to work: only flags the user
    actually typed override env/.env/defaults.
    """
    for field in fields:
        value = getattr(args, field, None)
        if value is not None:
            setattr(settings, field, value)
    return settings
