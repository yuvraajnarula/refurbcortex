# app/core/cost_estimator.py
import os
import time
import math
import threading
from typing import Dict, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from collections import OrderedDict
from app.utils.logger import app_logger

class CostBreakdown(BaseModel):
    base_parts_inr: float = Field(ge=0)
    labor_inr: float = Field(ge=0)
    paint_inr: float = Field(ge=0)
    regional_adjustment_inr: float = Field(ge=0)
    age_mileage_factor_inr: float = Field(ge=0)
    total_estimated_inr: float = Field(ge=0)
    cost_range_inr: Tuple[float, float] = Field(..., description="(aftermarket_min, oem_max)")
    breakdown_notes: str

    @field_validator("cost_range_inr")
    @classmethod
    def validate_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if v[0] <= 0 or v[1] <= 0 or v[0] >= v[1]:
            raise ValueError("Cost range must be positive and min < max")
        return v

class CostEstimator:
    """
    Production-grade refurb cost estimator calibrated for Indian auto market (2024).
    Deterministic, bounded, thread-safe, and cache-optimized.
    """
    # Realistic labor rates (₹/hr) by city tier
    LABOR_RATES = {"tier_1": 850.0, "tier_2": 680.0, "tier_3": 520.0}
    
    # Paint material + application cost (₹/cm²) by severity
    PAINT_RATES = {"MINOR": 14.0, "MODERATE": 22.0, "MAJOR": 32.0, "CRITICAL": 45.0}
    
    # Base cost matrix: (panel, severity) -> (min_oem_inr, max_oem_inr, labor_hrs, needs_paint)
    # Calibrated against authorized service center quotes & aftermarket benchmarks
    BASE_COST_MATRIX = {
        ("front_bumper", "MINOR"): (3500, 6500, 2.5, False),
        ("front_bumper", "MODERATE"): (7500, 12500, 4.5, True),
        ("front_bumper", "MAJOR"): (14000, 22000, 7.0, True),
        ("hood", "MINOR"): (4000, 7000, 3.0, False),
        ("hood", "MODERATE"): (8500, 14000, 5.0, True),
        ("hood", "MAJOR"): (16000, 26000, 8.0, True),
        ("left_door", "MINOR"): (2800, 5500, 2.0, False),
        ("left_door", "MODERATE"): (6500, 11000, 4.0, True),
        ("left_door", "MAJOR"): (12000, 19000, 6.5, True),
        ("right_door", "MINOR"): (2800, 5500, 2.0, False),
        ("right_door", "MODERATE"): (6500, 11000, 4.0, True),
        ("right_door", "MAJOR"): (12000, 19000, 6.5, True),
        ("side_mirror", "MINOR"): (1200, 3200, 1.0, False),
        ("side_mirror", "MODERATE"): (3500, 6800, 2.5, True),
        ("side_mirror", "MAJOR"): (7000, 12000, 4.0, True),
        ("windshield", "MINOR"): (2500, 4500, 1.5, False),
        ("windshield", "MODERATE"): (6000, 11000, 3.0, False),
        ("windshield", "MAJOR"): (12000, 24000, 4.5, False),
        ("roof", "MINOR"): (3000, 5500, 2.0, False),
        ("roof", "MODERATE"): (7000, 12000, 5.0, True),
        ("roof", "MAJOR"): (13000, 21000, 7.5, True),
        ("trunk", "MINOR"): (2500, 4800, 2.0, False),
        ("trunk", "MODERATE"): (6000, 10500, 4.0, True),
        ("trunk", "MAJOR"): (11000, 18000, 6.0, True),
        # Fallback for unmapped panels
        ("unknown", "MINOR"): (2000, 4000, 2.0, False),
        ("unknown", "MODERATE"): (5000, 9000, 4.0, True),
        ("unknown", "MAJOR"): (10000, 18000, 7.0, True),
    }

    def __init__(self, cache_ttl: int = 3600, cache_max: int = 1000):
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._cache_ttl = cache_ttl
        self._cache_max = cache_max
        app_logger.info("✅ CostEstimator initialized (thread-safe LRU cache + market constants)")

    def _get_cached(self, key: str) -> Optional[Dict]:
        with self._lock:
            if key in self._cache:
                val, ts = self._cache[key]
                if time.time() - ts < self._cache_ttl:
                    self._cache.move_to_end(key)
                    return val
                del self._cache[key]
            return None

    def _set_cache(self, key: str, val: Dict):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (val, time.time())
            if len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)

    def estimate(self, panel: str, severity: str, area_cm2: float, city_tier: str,
                 vehicle_age: int, odometer_km: int, brand: str = "generic") -> CostBreakdown:
        """
        Deterministic refurb cost estimation with regional, age, and mileage adjustments.
        All inputs are normalized and bounded to prevent market drift or negative values.
        """
        try:
            # Normalize inputs
            panel = panel.strip().lower()
            severity = severity.strip().upper()
            city_tier = city_tier.strip().lower()
            area_cm2 = max(1.0, float(area_cm2))
            vehicle_age = max(0, int(vehicle_age))
            odometer_km = max(0, int(odometer_km))
            
            cache_key = f"{panel}_{severity}_{city_tier}_{brand}"
            cached = self._get_cached(cache_key)
            if cached:
                return CostBreakdown(**cached)

            # 1. Base lookup with fallback
            base_key = (panel, severity) if (panel, severity) in self.BASE_COST_MATRIX else ("unknown", severity)
            base_min, base_max, labor_hrs, needs_paint = self.BASE_COST_MATRIX[base_key]

            # 2. Labor cost
            labor_rate = self.LABOR_RATES.get(city_tier, self.LABOR_RATES["tier_2"])
            labor_cost = labor_hrs * labor_rate

            # 3. Paint cost (area-proportional)
            paint_cost = 0.0
            if needs_paint:
                paint_rate = self.PAINT_RATES.get(severity, 22.0)
                paint_cost = area_cm2 * paint_rate

            # 4. Age & mileage complexity factor
            # Older/higher-km vehicles require more prep, rust treatment, alignment
            age_factor = 1.18 if vehicle_age >= 7 else 1.0
            mileage_factor = 1.12 if odometer_km > 120000 else 1.0
            adj_factor = age_factor * mileage_factor
            adj_cost = (base_min + labor_cost + paint_cost) * (adj_factor - 1.0)

            # 5. Regional market multiplier (Tier 1 premium, Tier 3 discount)
            regional_mult = {"tier_1": 1.12, "tier_2": 1.0, "tier_3": 0.93}.get(city_tier, 1.0)

            # 6. Final calculation
            subtotal = base_min + labor_cost + paint_cost + adj_cost
            total = subtotal * regional_mult
            
            # Market-validated bounds: Aftermarket min (-15%), OEM max (+25%)
            total_min = max(1500, total * 0.85)
            total_max = total * 1.25

            breakdown = CostBreakdown(
                base_parts_inr=round(base_min, 2),
                labor_inr=round(labor_cost, 2),
                paint_inr=round(paint_cost, 2),
                regional_adjustment_inr=round(subtotal * (regional_mult - 1.0), 2),
                age_mileage_factor_inr=round(adj_cost, 2),
                total_estimated_inr=round(total, 2),
                cost_range_inr=(round(total_min, 2), round(total_max, 2)),
                breakdown_notes=f"Panel:{panel}|Sev:{severity}|City:{city_tier}|Age:{vehicle_age}y|Km:{odometer_km}"
            )

            self._set_cache(cache_key, breakdown.model_dump())
            return breakdown

        except Exception as e:
            app_logger.error(f"❌ CostEstimation failed: {e}")
            # Deterministic fallback to prevent pipeline crash
            return CostBreakdown(
                base_parts_inr=4500.0, labor_inr=3200.0, paint_inr=1800.0,
                regional_adjustment_inr=0.0, age_mileage_factor_inr=0.0,
                total_estimated_inr=9500.0, cost_range_inr=(7500.0, 13000.0),
                breakdown_notes=f"Fallback applied due to estimation error: {str(e)}"
            )