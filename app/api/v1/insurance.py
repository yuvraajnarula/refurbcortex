# app/api/v1/insurance.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from app.core.security import verify_api_key
from app.services.insurance_claim import map_to_irdai, compute_fraud_score
from app.utils.logger import app_logger
import time, uuid

router = APIRouter()

class ClaimRequest(BaseModel):
    inspection_id: str
    vin: Optional[str] = None
    grader_id: str
    human_approved: bool = True

@router.post("/claim")
async def submit_claim(req: ClaimRequest, api_key: str = Depends(verify_api_key)):
    if not req.human_approved:
        raise HTTPException(403, "Claim requires human approval (HITL override must be logged)")
    
    # In prod: fetch inspection from DB/cache. For P2: mock payload retrieval
    mock_inspection = {
        "inspection_id": req.inspection_id,
        "vin": req.vin or "UNKNOWN",
        "summary": {"total_adjusted_cost_inr": 12500.0},
        "routing": {"confidence": 0.92},
        "damage_records": [
            {"panel_affected": "front_bumper", "severity_label": "MODERATE", "repair_cost_max_inr": 8000},
            {"panel_affected": "hood", "severity_label": "MINOR", "repair_cost_max_inr": 4500}
        ],
        "hitl_overridden": False,
        "override_history": {"override_rate": 0.1}
    }
    
    try:
        claim_id = f"CLM_{uuid.uuid4().hex[:8].upper()}"
        irdai_claim = map_to_irdai(mock_inspection, claim_id)
        app_logger.info(f"📋 Insurance claim generated: {claim_id} | FraudScore: {irdai_claim.fraud_score}")
        
        return {
            "status": "success",
            "claim_id": claim_id,
            "irdai_payload": irdai_claim.model_dump(),
            "sync_status": "PENDING_WEBHOOK"
        }
    except Exception as e:
        app_logger.error(f"❌ Claim generation failed: {e}")
        raise HTTPException(500, detail="Insurance mapping failed")