from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.feedback_loop import FeedbackLoop
from app.utils.logger import app_logger

router = APIRouter()
feedback_engine = FeedbackLoop()

class FeedbackRequest(BaseModel):
    inspection_id: str
    predicted_cost_inr: float
    actual_cost_inr: float
    human_override: bool = False
    notes: str = ""

@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    try:
        result = feedback_engine.record(
            inspection_id=req.inspection_id,
            predicted_cost=req.predicted_cost_inr,
            actual_cost=req.actual_cost_inr,
            human_override=req.human_override,
            notes=req.notes
        )
        return {"status": "success", "data": result}
    except Exception as e:
        app_logger.error(f"❌ /feedback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))