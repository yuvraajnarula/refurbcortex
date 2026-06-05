import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from app.utils.logger import app_logger

class InspectionDataLoader:
    def __init__(self, csv_path: str = "./data/inspections.csv"):
        self.csv_path = Path(csv_path)
        self.df: Optional[pd.DataFrame] = None
        self._load()
    
    def _load(self):
        if not self.csv_path.exists():
            app_logger.warning(f"CSV not found: {self.csv_path}. Using empty schema.")
            self.df = pd.DataFrame(columns=self._get_expected_columns())
            return
        try:
            self.df = pd.read_csv(self.csv_path)
            app_logger.info(f"Loaded {len(self.df)} inspection records")
        except Exception as e:
            app_logger.error(f"Failed to load CSV: {e}")
            self.df = pd.DataFrame(columns=self._get_expected_columns())
    
    def _get_expected_columns(self) -> List[str]:
        return [
            "inspection_id","vehicle_type","vehicle_brand","vehicle_model",
            "manufacture_year","vehicle_age","fuel_type","transmission",
            "odometer_km","ownership_count","city_tier","city",
            "panel_affected","damage_category","damage_subtype","damage_cause",
            "severity_label","severity_score","damage_area_cm2",
            "repair_method","repair_cost_min_inr","repair_cost_max_inr",
            "labor_hours","paint_required","parts_replacement",
            "safety_risk","drivable_status","weather_condition",
            "lighting_condition","camera_device","service_center_type",
            "insurance_type","historical_claims","flood_damage",
            "rust_probability","structural_integrity_score",
            "battery_health_impact","ai_confidence","inspection_confidence",
            "human_override","recommended_action",
            "predicted_resale_impact_inr","expected_profit_after_repair_inr",
            "fraud_suspected","agent_consensus_score","repair_priority"
        ]
    
    def get_repair_cost_range(self, panel: str, category: str, severity: str) -> Dict[str, float]:
        """Lookup min/max cost from historical data"""
        if self.df is None or self.df.empty:
            return {"min": 5000, "max": 15000, "median": 10000}
        
        subset = self.df[
            (self.df["panel_affected"] == panel) &
            (self.df["damage_category"] == category) &
            (self.df["severity_label"] == severity)
        ]
        if subset.empty:
            subset = self.df[self.df["damage_category"] == category]
        
        if subset.empty:
            return {"min": 5000, "max": 15000, "median": 10000}
        
        return {
            "min": float(subset["repair_cost_min_inr"].median()),
            "max": float(subset["repair_cost_max_inr"].median()),
            "median": float(subset[["repair_cost_min_inr", "repair_cost_max_inr"]].mean(axis=1).median())
        }
    
    def get_panel_repair_methods(self, panel: str) -> List[str]:
        """Get common repair methods for a panel"""
        if self.df is None:
            return ["touchup_paint", "panel_repair", "replacement"]
        methods = self.df[self.df["panel_affected"] == panel]["repair_method"].dropna().unique()
        return methods.tolist() if len(methods) > 0 else ["touchup_paint"]