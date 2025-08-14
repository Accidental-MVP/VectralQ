from __future__ import annotations

import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import request_id_var, tenant_id_var
from app.core.config import get_settings


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request_id_var.set(request_id)

        tenant_header = get_settings().tenant_header
        tenant_id = request.headers.get(tenant_header)
        if tenant_id is None or tenant_id.strip() == "":
            # Missing tenant header: reject early
            return Response(status_code=400, content=b"Missing X-Tenant-ID header")

        tenant_id_var.set(tenant_id)
        try:
            response = await call_next(request)
        finally:
            # Clear after request completes
            request_id_var.set(None)
            tenant_id_var.set(None)
        # Propagate request id back
        response.headers.setdefault("X-Request-ID", request_id)
        return response


