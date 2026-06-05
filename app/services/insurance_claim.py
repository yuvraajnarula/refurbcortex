from pydantic import BaseModel, Field
from typing import List, Optional
import hashlib, time
from app.utils.logger import app_logger

class IRDAIClaim(BaseModel):
    claim_id: str
    inspection_id: str
    vehicle_reg_hash: str 
    damage_summary: str
    estimated_cost_inr: float
    ai_confidence: float
    human_override: bool = False
    fraud_score: float = Field(ge=0.0, le=1.0)
    audit_trail_url: Optional[str] = None

def compute_fraud_score(records: List[dict], override_history: dict) -> float:
    score = 0.0
    if len(records) > 5: score += 0.2  # Excessive damage flags
    costs = [r["repair_cost_max_inr"] for r in records]
    if max(costs) > 3 * (sum(costs)/max(len(costs),1)): score += 0.3  # Cost skew
    if override_history.get("override_rate", 0) > 0.4: score += 0.3  # High override ratio
    return min(1.0, round(score, 2))

def map_to_irdai(inspection: dict, claim_id: str) -> IRDAIClaim:
    return IRDAIClaim(
        claim_id=claim_id,
        inspection_id=inspection["inspection_id"],
        vehicle_reg_hash=hashlib.sha256(inspection.get("vin","").encode()).hexdigest()[:12],
        damage_summary=", ".join([f"{r['panel_affected']} ({r['severity_label']})" for r in inspection.get("damage_records",[])]),
        estimated_cost_inr=inspection["summary"]["total_adjusted_cost_inr"],
        ai_confidence=inspection["routing"]["confidence"],
        human_override=inspection.get("hitl_overridden", False),
        fraud_score=compute_fraud_score(inspection.get("damage_records",[]), inspection.get("override_history", {}))
    )