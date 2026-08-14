"""Auth middleware."""

def require_bearer(header: str) -> str:
    if not header.startswith("Bearer "):
        raise PermissionError("missing bearer")
    return header[7:]
