"""ASGI entry point for the demo server.

Run from the repo root:

    uvicorn app.backend.main:app --reload

The app object is built at import time by `create_app()`, which is what
uvicorn's reloader needs (it re-imports this module, it does not call a
factory). Every path the app touches is resolved from the environment by
`config.get_settings()`, so this module reads no configuration itself.
"""

from __future__ import annotations

from .api import create_app

app = create_app()
