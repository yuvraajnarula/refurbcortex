from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib
import time
from collections import OrderedDict

_idempotency_cache = OrderedDict()
CACHE_TTL_SECONDS = 300
MAX_CACHE_SIZE = 1000

class IdempotencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method != "POST":
            return await call_next(request)

        idem_key = request.headers.get("X-Idempotency-Key")
        if not idem_key:
            return await call_next(request)

        if idem_key in _idempotency_cache:
            cached_time, cached_response = _idempotency_cache[idem_key]
            if time.time() - cached_time < CACHE_TTL_SECONDS:
                return cached_response
            else:
                del _idempotency_cache[idem_key]

        response = await call_next(request)

        if len(_idempotency_cache) > MAX_CACHE_SIZE:
            _idempotency_cache.popitem(last=False)
        _idempotency_cache[idem_key] = (time.time(), response)

        return response