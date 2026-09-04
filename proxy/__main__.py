"""
Entry point for the proxy, so that it can be started with `uv run -m proxy`.
"""

from __future__ import annotations

from proxy.app import app

if __name__ == "__main__":
    app()
