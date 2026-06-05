import numpy as np
from app.utils.logger import app_logger

class DigitalTwinForecaster:
    DECAY_CURVES = {
        "battery_soh": lambda soh, km: max(40, soh - (km * 0.0008)),
        "paint_integrity": lambda score, km: max(0, score - (km * 0.0012)),
        "suspension_life": lambda life, km: max(0, life - (km * 0.0005)),
        "brake_pad_thickness": lambda mm, km: max(2.0, mm - (km * 0.00003))
    }

    def simulate(self, current_state: dict, forward_km: int = 20000) -> dict:
        projection = {}
        for comp, decay_fn in self.DECAY_CURVES.items():
            val = current_state.get(comp, 100.0)
            projection[comp] = {
                "current": val,
                "at_target_km": round(decay_fn(val, forward_km), 1),
                "risk_flag": decay_fn(val, forward_km) < self._get_threshold(comp)
            }
        return projection

    def _get_threshold(self, comp: str) -> float:
        return {"battery_soh": 60, "paint_integrity": 30, "suspension_life": 40, "brake_pad_thickness": 3.0}[comp]