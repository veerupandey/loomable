"""API Tool runtime for the loomable agent framework.

Implements the APITool class that sends configured HTTP requests to external
services and returns responses as ToolResults. Handles non-success HTTP statuses
and request timeouts per the framework error taxonomy.
"""

from __future__ import annotations

from typing import Any

import httpx

from loomable.kernel.contracts import Tool
from loomable.kernel.errors import APIToolError, APIToolTimeoutError
from loomable.kernel.models import APIToolSpec, ToolResult


class APITool(Tool):
    """A tool that invokes an external service over HTTP.

    Configured via an APIToolSpec dict containing:
      - name: tool name
      - description: tool description
      - method: HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)
      - url: target URL (may contain placeholders for arg substitution)
      - headers: optional dict of HTTP headers
      - timeout: request timeout in seconds
    """

    def __init__(self, spec: APIToolSpec) -> None:
        self.spec = spec
        self.name: str = spec["name"]
        self.description: str = spec.get("description", "")

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        """Send the configured HTTP request and return the response.

        Args:
            args: Arguments for the request. May include 'body' for request
                  payload, 'params' for query parameters, and 'headers' for
                  per-request header overrides.

        Returns:
            ToolResult with the response content and metadata.

        Raises:
            APIToolError: When the HTTP response has a non-success status code.
            APIToolTimeoutError: When the request exceeds the configured timeout.
        """
        method: str = self.spec["method"]
        url: str = self.spec["url"]
        headers: dict[str, str] = dict(self.spec.get("headers", {}))
        timeout: float = self.spec.get("timeout", 30.0)

        # Merge per-request headers if provided
        if "headers" in args:
            headers.update(args["headers"])

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "method": method,
            "url": url,
            "headers": headers,
            "timeout": timeout,
        }

        # Add body/params from args
        if "body" in args:
            request_kwargs["json"] = args["body"]
        if "params" in args:
            request_kwargs["params"] = args["params"]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(**request_kwargs)
        except httpx.TimeoutException:
            raise APIToolTimeoutError(tool_name=self.name, timeout=timeout)

        if not response.is_success:
            raise APIToolError(status_code=response.status_code)

        # Determine content type for response parsing
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            content = response.json()
        else:
            content = response.text

        return ToolResult(
            content=content,
            metadata={
                "status_code": response.status_code,
                "headers": dict(response.headers),
            },
        )
