from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, List
from datetime import datetime
from enum import Enum

class SeverityLabel(str, Enum):
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"

class DamageCategory(str, Enum):
    SCRATCH = "scratch"
    DENT = "dent"
    CORROSION = "corrosion"
    ELECTRICAL = "electrical"
    STRUCTURAL = "structural"
    GLASS = "glass"
    TIRE = "tire"
    OTHER = "other"

class RecommendedAction(str, Enum):
    REFURBISH = "REFURBISH"
    SELL_AS_IS = "SELL_AS_IS"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"
    REJECT = "REJECT"

class RepairPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class InspectionInput(BaseModel):
    image_b64: str 
    vehicle_type: str = "car"
    vehicle_brand: str
    vehicle_model: str
    manufacture_year: int
    fuel_type: str
    transmission: str
    odometer_km: int
    ownership_count: int
    city: str
    city_tier: Literal["tier_1", "tier_2", "tier_3"]
    panel_affected: str
    damage_category: DamageCategory
    damage_subtype: Optional[str] = None
    lighting_condition: Optional[str] = None
    weather_condition: Optional[str] = None

class DamageRecord(BaseModel):
    inspection_id: Optional[str] = None
    panel_affected: str
    damage_category: DamageCategory
    damage_subtype: Optional[str]
    severity_label: SeverityLabel
    severity_score: float = Field(ge=0, le=10)
    damage_area_cm2: float
    repair_method: str
    repair_cost_min_inr: float
    repair_cost_max_inr: float
    labor_hours: float
    paint_required: bool
    parts_replacement: bool
    safety_risk: Literal["low", "medium", "high", "critical"]
    drivable_status: bool
    rust_probability: float = Field(ge=0, le=1)
    structural_integrity_score: float = Field(ge=0, le=1)
    battery_health_impact: float = Field(ge=0, le=1)
    ai_confidence: float = Field(ge=0, le=1)
    inspection_confidence: float = Field(ge=0, le=1)
    recommended_action: RecommendedAction
    predicted_resale_impact_inr: float
    expected_profit_after_repair_inr: float
    repair_priority: RepairPriority
    agent_consensus_score: float = Field(ge=0, le=1)
    timestamp: datetime = Field(default_factory=datetime.now)

class RoutingDecision(BaseModel):
    status: Literal["CONFIDENT", "UNCERTAIN"]
    confidence: float
    action: str
    requires_human_review: bool


class AgentRecommendation(BaseModel):
    recommendation: Literal["REPAIR", "AS_IS", "PARTIAL"]
    reasoning: str
    expected_net_margin_pct: float
    turnover_risk: Literal["LOW", "MED", "HIGH"]
    repair_priority_items: List[str] = []

class InspectionResponse(BaseModel):
    status: Literal["success", "error", "uncertain"]
    inspection_id: str
    heatmap_b64: Optional[str] = None
    damage_records: List[DamageRecord]
    summary: dict
    routing: RoutingDecision
    recommendation: Optional[AgentRecommendation] = None
    meta: dict

class SHAPDriver(BaseModel):
    feature: str
    input_value: float
    contribution_inr: float
    impact: Literal["increases cost", "decreases cost"]

class SHAPBreakdown(BaseModel):
    base_estimate_inr: float
    xai_adjusted_cost_inr: float
    top_drivers: List[SHAPDriver]


class InspectionResponse(BaseModel):
    status: Literal["success", "error", "uncertain"]
    inspection_id: str
    heatmap_b64: Optional[str] = None
    damage_records: List[DamageRecord]
    summary: dict
    routing: RoutingDecision
    recommendation: Optional[AgentRecommendation] = None
    shap_breakdown: Optional[SHAPBreakdown] = None
    meta: dict
    