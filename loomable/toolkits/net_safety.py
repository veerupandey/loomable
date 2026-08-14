"""Shared HTTP / SSRF safety helpers for research toolkits."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

__all__ = [
    "is_blocked_host",
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


def is_blocked_host(hostname: str) -> bool:
    """Return True when *hostname* must not be fetched (SSRF guard).

    DNS resolution failures fail closed (treat as blocked) so attackers cannot
    bypass the guard with unresolvable or flaky names.
    """
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return True
    if host in _BLOCKED_LITERALS or host.endswith(".localhost"):
        return True
    # Literal IP forms
    try:
        ip = ipaddress.ip_address(host)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True  # fail closed
    if not infos:
        return True
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    return False


def validate_http_url(url: str, *, block_private_hosts: bool = True) -> str | None:
    """Return an error string when *url* is unsafe, else ``None``."""
    raw = (url or "").strip()
    if not raw:
        return "Error: url is required"
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        return f"Error: unsupported URL scheme: {parsed.scheme or '(none)'}"
    host = parsed.hostname or ""
    if block_private_hosts and is_blocked_host(host):
        return f"Error: blocked host (SSRF guard): {host or '(none)'}"
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
