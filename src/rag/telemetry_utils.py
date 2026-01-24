"""
Utilities to disable optional telemetry safely.
"""
from __future__ import annotations

import os


def disable_chroma_telemetry() -> None:
    """Best-effort kill-switch for Chroma telemetry emission."""
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
    os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")

    try:
        import posthog  # type: ignore
    except Exception:
        return

    def _noop_capture(*_args, **_kwargs) -> None:
        return None

    try:
        posthog.capture = _noop_capture  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        client = getattr(posthog, "client", None)
        if client is not None and hasattr(client, "capture"):
            client.capture = _noop_capture  # type: ignore[attr-defined]
    except Exception:
        pass
