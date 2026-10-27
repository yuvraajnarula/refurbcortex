import pytest
from app.api.v1.schema import DamageRecord, SeverityLabel, DamageCategory, RecommendedAction, RepairPriority


def test_damage_record_accepts_bbox():
    r = DamageRecord(
        panel_affected="front_bumper",
        damage_category=DamageCategory.DENT,
        damage_subtype=None,
        severity_label=SeverityLabel.MINOR,
        severity_score=2.5,
        damage_area_cm2=10.0,
        repair_method="touchup_paint",
        repair_cost_min_inr=1000.0,
        repair_cost_max_inr=2000.0,
        labor_hours=1.0,
        paint_required=False,
        parts_replacement=False,
        safety_risk="low",
        drivable_status=True,
        rust_probability=0.1,
        structural_integrity_score=0.9,
        battery_health_impact=0.0,
        ai_confidence=0.8,
        inspection_confidence=0.76,
        recommended_action=RecommendedAction.REFURBISH,
        predicted_resale_impact_inr=-100.0,
        expected_profit_after_repair_inr=500.0,
        repair_priority=RepairPriority.LOW,
        agent_consensus_score=0.75,
    )
    r.bbox = [1.0, 2.0, 3.0, 4.0]
    assert r.bbox == [1.0, 2.0, 3.0, 4.0]
    assert r.model_dump()["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_damage_record_severity_bounds():
    with pytest.raises(Exception):
        DamageRecord(
            panel_affected="hood",
            damage_category=DamageCategory.SCRATCH,
            damage_subtype=None,
            severity_label=SeverityLabel.MINOR,
            severity_score=11.0,  # > 10 invalid
            damage_area_cm2=10.0,
            repair_method="touchup_paint",
            repair_cost_min_inr=1000.0,
            repair_cost_max_inr=2000.0,
            labor_hours=1.0,
            paint_required=False,
            parts_replacement=False,
            safety_risk="low",
            drivable_status=True,
            rust_probability=0.1,
            structural_integrity_score=0.9,
            battery_health_impact=0.0,
            ai_confidence=0.8,
            inspection_confidence=0.76,
            recommended_action=RecommendedAction.REFURBISH,
            predicted_resale_impact_inr=-100.0,
            expected_profit_after_repair_inr=500.0,
            repair_priority=RepairPriority.LOW,
            agent_consensus_score=0.75,
        )
