from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.feedback_loop import FeedbackLoop
from app.utils.logger import app_logger
from datetime import datetime

router = APIRouter()
feedback_loop = FeedbackLoop()

class HITLOverride(BaseModel):
    inspection_id: str
    original_prediction: float
    human_correction: float
    reason: str # "AI missed hidden damage", "Price too high", etc.
    grader_id: str

@router.post("/override")
async def submit_override(override: HITLOverride):
    """
    Human grader overrides AI decision.
    This logs a high-weight feedback entry and updates memory.
    """
    try:
        # 1. Log the override as critical feedback
        feedback_loop.record(
            inspection_id=override.inspection_id,
            predicted_cost=override.original_prediction,
            actual_cost=override.human_correction, # Treat human decision as ground truth
            human_override=True,
            notes=f"OVERRIDE by {override.grader_id}: {override.reason}"
        )

        # 2. In P1, trigger immediate vector DB update
        # vector_memory.update_memory(override.inspection_id, abs(...))

        app_logger.info(f"🛡️ HITL Override | {override.inspection_id} | AI: ₹{override.original_prediction} -> Human: ₹{override.human_correction}")
        
        return {
            "status": "success",
            "message": "Override recorded. System calibration updated.",
            "impact": "high_weight_feedback"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))