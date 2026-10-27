import pytest
from app.api.v1.webhooks import _status_db


def test_status_store_memory_set_get():
    _status_db["INS_TEST1"] = {"inspection_id": "INS_TEST1", "status": "PROCESSING"}
    assert "INS_TEST1" in _status_db
    assert _status_db["INS_TEST1"]["status"] == "PROCESSING"


def test_status_store_memory_missing_returns_none_on_aget_sync():
    # aget is async; fallback dict __getitem__ raises KeyError if missing — use .get
    assert _status_db._fallback.get("INS_DOES_NOT_EXIST") is None


@pytest.mark.asyncio
async def test_status_store_async_roundtrip():
    await _status_db.aset("INS_ASYNC", {
        "inspection_id": "INS_ASYNC",
        "status": "COMPLETED",
        "message": "ok",
    })
    val = await _status_db.aget("INS_ASYNC")
    assert val is not None
    assert val["status"] == "COMPLETED"
    assert val["inspection_id"] == "INS_ASYNC"


@pytest.mark.asyncio
async def test_status_store_async_missing():
    val = await _status_db.aget("INS_DOES_NOT_EXIST_ASYNC")
    assert val is None
