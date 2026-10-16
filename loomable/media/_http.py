"""HTTP destination validation shared by media downloads and research tools."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin

__all__ = [
    "is_blocked_host",
    "pinned_http_target",
    "validate_http_url",
    "validate_redirect_target",
]

_BLOCKED_LITERALS = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata.google",
        "0.0.0.0",
        "::",
        "::1",
    }
)


def _public_addresses(hostname: str) -> list[str]:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host or host in _BLOCKED_LITERALS or host.endswith(".localhost"):
        return []
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addresses = [ipaddress.ip_address(info[4][0]) for info in socket.getaddrinfo(host, None)]
        except (OSError, ValueError):
            return []
    if any(not ip.is_global or ip.is_multicast or ip.is_reserved or ip.is_unspecified
           for ip in addresses):
        return []
    return [str(ip) for ip in sorted(addresses, key=lambda ip: ip.version)]


def is_blocked_host(hostname: str) -> bool:
    """Fail closed when DNS is unavailable or any address is non-public."""
    return not _public_addresses(hostname)


def pinned_http_target(url: str, *, block_private_hosts: bool = True):
    """Return the validated HTTPX URL and parameters for its connection."""
    import httpx

    raw = (url or "").strip()
    if not raw:
        raise ValueError("url is required")
    target = httpx.URL(raw)
    if target.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {target.scheme or '(none)'}")
    if not target.host or target.userinfo:
        raise ValueError("URL requires a host and cannot contain credentials")
    if not block_private_hosts:
        return target, {}
    host = target.raw_host.decode("ascii")
    addresses = _public_addresses(host)
    if not addresses:
        raise ValueError(f"blocked host (SSRF guard): {host}")
    return target.copy_with(host=addresses[0]), {
        "headers": {"Host": target.netloc.decode("ascii")},
        "extensions": {"sni_hostname": host},
    }


def validate_http_url(url: str, *, block_private_hosts: bool = True) -> str | None:
    """Return an error string when *url* is unsafe, else ``None``."""
    import httpx

    try:
        pinned_http_target(url, block_private_hosts=block_private_hosts)
    except (ValueError, httpx.InvalidURL) as exc:
        return f"Error: {exc}"
    return None


def validate_redirect_target(
    base_url: str,
    location: str,
    *,
    block_private_hosts: bool = True,
) -> tuple[str | None, str | None]:
    """Resolve a redirect Location against *base_url*.

    Returns ``(error, absolute_url)``.
    """
    loc = (location or "").strip()
    if not loc:
        return "Error: redirect missing Location header", None
    absolute = urljoin(base_url, loc)
    err = validate_http_url(absolute, block_private_hosts=block_private_hosts)
    if err:
        return err, None
    return None, absolute
