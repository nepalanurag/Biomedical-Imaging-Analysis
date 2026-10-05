"""Structured logging for the pipeline.

One setup call configures the root logger; every module then uses
``get_logger(__name__)``. Records go to stderr as JSON lines (default, for
log aggregation) or human-readable text (``CTPIPE_LOG_FORMAT=text``).

Context fields are keyword arguments:

    logger.info("series_ingested", series_uid=uid, slices=n)

Use ``bind`` to attach fixed context to a logger:

    slog = bind(logger, series_uid=uid)
    slog.info("series_quantified", infection_pct=3.2)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_ctx_"):
                payload[key[5:]] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    _configured = True


class _StructuredLogger:
    """Thin wrapper: kwargs become structured context fields on the record."""

    def __init__(self, logger: logging.Logger, ctx: dict | None = None):
        self._logger = logger
        self._ctx = dict(ctx or {})

    def _call(self, method: str, msg: str, *args, **kwargs):
        exc_info = kwargs.pop("exc_info", None)
        extra = {f"_ctx_{k}": v for k, v in {**self._ctx, **kwargs}.items()}
        getattr(self._logger, method)(msg, *args, extra=extra, exc_info=exc_info)

    def debug(self, msg: str, *args, **kwargs):
        self._call("debug", msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._call("info", msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._call("warning", msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._call("error", msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        kwargs["exc_info"] = True
        self._call("error", msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._call("critical", msg, *args, **kwargs)


def get_logger(name: str) -> _StructuredLogger:
    return _StructuredLogger(logging.getLogger(name))


def bind(logger: _StructuredLogger, **ctx) -> _StructuredLogger:
    """Return a logger with fixed context fields attached to every record."""
    if isinstance(logger, _StructuredLogger):
        return _StructuredLogger(logger._logger, {**logger._ctx, **ctx})
    return _StructuredLogger(logger, ctx)
