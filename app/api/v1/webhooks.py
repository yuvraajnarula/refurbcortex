from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

router = APIRouter()
_status_db = {}

class StatusUpdate(BaseModel):
    inspection_id: str
    status: str
    message: Optional[str] = None

@router.get("/status/{inspection_id}")
async def get_status(inspection_id: str):
    if inspection_id not in _status_db:
        raise HTTPException(404, "Inspection not found")
    return _status_db[inspection_id]

@router.post("/status")
async def update_status(payload: StatusUpdate):
    _status_db[payload.inspection_id] = {
        "inspection_id": payload.inspection_id,
        "status": payload.status,
        "message": payload.message,
        "ts": time.time()
    }
    return {"status": "success"}