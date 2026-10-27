from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib
import time
import json
from app.config import settings
from app.utils.logger import app_logger

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except Exception:
    _REDIS_AVAILABLE = False

CACHE_TTL_SECONDS = 300
_PREFIX = "idem:"

class IdempotencyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_client=None):
        super().__init__(app)
        self.redis = redis_client
        self._redis_ready = None  # None=unknown, True/False after first check

    async def _get_redis(self):
        if not _REDIS_AVAILABLE:
            return None
        if self.redis is not None:
            return self.redis
        if self._redis_ready is False:
            return None
        try:
            r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            await r.ping()
            self._redis_ready = True
            app_logger.info("Idempotency middleware backed by Redis")
            return r
        except Exception as e:
            app_logger.warning(f"Redis unavailable for idempotency, using memory fallback: {e}")
            self._redis_ready = False
            return None

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST":
            return await call_next(request)

        idem_key = request.headers.get("X-Idempotency-Key")
        if not idem_key:
            return await call_next(request)

        cache_key = f"{_PREFIX}{idem_key}"

        # Cross-worker dedup via Redis
        r = await self._get_redis()
        if r is not None:
            try:
                if await r.exists(cache_key):
                    return Response(
                        content=json.dumps({"detail": "Duplicate request", "idempotency_key": idem_key}),
                        status_code=409,
                        media_type="application/json",
                    )
                await r.set(cache_key, "1", ex=CACHE_TTL_SECONDS)
            except Exception as e:
                app_logger.warning(f"Redis idempotency check failed: {e}")

        response = await call_next(request)
        return response
