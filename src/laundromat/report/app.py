"""Shim so the Dockerfile CMD (uvicorn laundromat.report.app:app) and
uvicorn laundromat.report:app both serve the same instance."""

from laundromat.report import app  # noqa: F401
