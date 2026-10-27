from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json, time, asyncio
from app.config import settings
from app.utils.logger import app_logger

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except Exception:
    _REDIS_AVAILABLE = False

router = APIRouter()

_PREFIX = "inspect:status:"

class StatusStore:
    """In-memory status store mirrored to Redis when available."""
    def __init__(self):
        self._fallback: dict = {}
        self._redis = None
        self._redis_ready = None  # None=unknown, True/False after first check

    async def _get_redis(self):
        if not _REDIS_AVAILABLE:
            return None
        if self._redis is not None:
            return self._redis
        if self._redis_ready is False:
            return None
        try:
            self._redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            await self._redis.ping()
            self._redis_ready = True
            app_logger.info("Status store backed by Redis")
            return self._redis
        except Exception as e:
            app_logger.warning(f"Redis unavailable for status store, using memory fallback: {e}")
            self._redis_ready = False
            self._redis = None
            return None

    async def _redis_set(self, key: str, value: dict, ttl: int = None):
        r = await self._get_redis()
        if r is None:
            return
        ttl = ttl or settings.REDIS_STATUS_TTL_SECONDS
        try:
            await r.set(f"{_PREFIX}{key}", json.dumps(value), ex=ttl)
        except Exception:
            pass

    async def _redis_get(self, key: str) -> Optional[dict]:
        r = await self._get_redis()
        if r is None:
            return None
        try:
            raw = await r.get(f"{_PREFIX}{key}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return None

    def __contains__(self, key: str) -> bool:
        return key in self._fallback

    def __getitem__(self, key: str) -> dict:
        return self._fallback[key]

    def __setitem__(self, key: str, value: dict):
        self._fallback[key] = value
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._redis_set(key, value))
        except RuntimeError:
            pass

    async def aget(self, key: str) -> Optional[dict]:
        val = await self._redis_get(key)
        if val is not None:
            return val
        return self._fallback.get(key)

    async def aset(self, key: str, value: dict, ttl: int = None):
        self._fallback[key] = value
        await self._redis_set(key, value, ttl=ttl)

_status_db = StatusStore()

class StatusUpdate(BaseModel):
    inspection_id: str
    status: str
    message: Optional[str] = None

@router.get("/status/{inspection_id}")
async def get_status(inspection_id: str):
    val = await _status_db.aget(inspection_id)
    if val is None:
        raise HTTPException(404, "Inspection not found")
    return val

@router.post("/status")
async def update_status(payload: StatusUpdate):
    await _status_db.aset(payload.inspection_id, {
        "inspection_id": payload.inspection_id,
        "status": payload.status,
        "message": payload.message,
        "ts": time.time()
    })
    return {"status": "success"}
