"""Keep the server's administrative routes local, even when inference is shared."""

import ipaddress
import os
import secrets
import urllib.request
from urllib.parse import urlsplit, urlunsplit

from fastapi import Request


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Internal requests can carry management credentials or inference data.
        return None


def open_local_http(
    request: urllib.request.Request, timeout: float | None = None, *,
    allow_non_loopback: bool = False,
):
    """Keep internal requests off configured proxies and reject redirects."""
    parsed = urlsplit(request.full_url)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Internal HTTP URLs cannot contain credentials")
    if parsed.hostname and parsed.hostname.lower() == "localhost":
        # Never rely on a name-service lookup for a request carrying secrets.
        authority = "127.0.0.1" + (f":{parsed.port}" if parsed.port is not None else "")
        request.full_url = urlunsplit(parsed._replace(netloc=authority))
        parsed = urlsplit(request.full_url)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        loopback = False
    if parsed.scheme != "http" or not parsed.hostname or (not loopback and not allow_non_loopback):
        raise ValueError("Internal HTTP requests require a loopback address")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    return opener.open(request, timeout=timeout)


def is_local_management_request(request: Request) -> bool:
    """Require a loopback peer and host, and reject browser-origin requests.

    Host and Origin checks also prevent a website from using DNS rebinding or a
    cross-origin form to exercise a local administrative endpoint.
    """
    try:
        if request.client is None or not ipaddress.ip_address(request.client.host).is_loopback:
            return False
    except ValueError:
        return False
    host = request.url.hostname
    if host is None:
        return False
    try:
        loopback_host = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback_host = host.lower() == "localhost"
    return loopback_host and "origin" not in request.headers


def is_management_path(path: str) -> bool:
    return (
        path in {"/api/pull", "/api/delete", "/api/show", "/v1/models/cache", "/v1/models/unload"}
        or (path.startswith("/v1/models/") and path.endswith("/unload"))
        or path.startswith("/internal/")
        or path.startswith("/management/")
    )


def is_management_authorized(request: Request) -> bool:
    """App-owned servers require a per-launch secret in addition to loopback."""
    token = os.environ.get("MLX_SERVE_MANAGEMENT_TOKEN")
    return not token or secrets.compare_digest(
        request.headers.get("authorization", ""), f"Bearer {token}"
    )


def internal_management_headers() -> dict[str, str]:
    token = os.environ.get("MLX_SERVE_MANAGEMENT_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}
