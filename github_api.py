"""Thin GitHub REST API client.

Single responsibility: make the HTTP call and either return parsed JSON
or raise a typed exception.
"""

from __future__ import annotations

import os
from typing import Any

import requests

GITHUB_API = "https://api.github.com"
HTTP_TIMEOUT = 20
USER_AGENT = "cloudbees-assessment-agent"

def gh_get(path: str, params: dict | None = None) -> Any:
    """GET against the GitHub REST API.

    Args:
        path: Path under https://api.github.com, e.g. "/repos/owner/name/readme".
        params: Optional query string params.

    Returns:
        Parsed JSON on HTTP 200.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(
        f"{GITHUB_API}{path}", params=params, headers=headers, timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()