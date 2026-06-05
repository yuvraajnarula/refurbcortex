import os
import pandas as pd
import numpy as np
import shap
import joblib
from sklearn.ensemble import RandomForestRegressor
from app.utils.logger import app_logger

class XAIEngine:
    def __init__(self, csv_path: str = "./data/inspections.csv", model_path: str = "./models/cost_xai_model.pkl"):
        self.csv_path = csv_path
        self.model_path = model_path
        self.feature_cols = ["severity_score", "damage_area_cm2", "labor_hours", "rust_probability", "structural_integrity_score"]
        self.model = self._load_or_train()
        self.explainer = shap.TreeExplainer(self.model)
        app_logger.info("XAI Engine initialized (SHAP-ready)")

    def _load_or_train(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        
        if not os.path.exists(self.csv_path):
            app_logger.warning("⚠️ CSV not found. Using fallback model for demo.")
            model = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42)
            dummy_X = pd.DataFrame(np.random.rand(100, len(self.feature_cols)), columns=self.feature_cols)
            dummy_y = dummy_X["severity_score"] * 3000 + dummy_X["labor_hours"] * 800 + 2000
            model.fit(dummy_X, dummy_y)
            joblib.dump(model, self.model_path)
            return model

        df = pd.read_csv(self.csv_path)
        df["median_cost"] = (df["repair_cost_min_inr"] + df["repair_cost_max_inr"]) / 2
        df = df[self.feature_cols + ["median_cost"]].dropna()
        X = df[self.feature_cols]
        y = df["median_cost"]
        
        model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        model.fit(X, y)
        joblib.dump(model, self.model_path)
        return model

    def generate_breakdown(self, damage_record: dict) -> dict:
        x = pd.DataFrame([{col: damage_record.get(col, 0.0) for col in self.feature_cols}])
        shap_vals = self.explainer.shap_values(x)[0]  # (1, n_features) -> array
        base = float(self.explainer.expected_value)
        total = base + float(shap_vals.sum())

        drivers = []
        for i, col in enumerate(self.feature_cols):
            val = float(shap_vals[i])
            drivers.append({
                "feature": col.replace("_", " ").title(),
                "input_value": x.iloc[0][col],
                "contribution_inr": round(val, 2),
                "impact": "increases cost" if val > 0 else "decreases cost"
            })
        drivers.sort(key=lambda d: abs(d["contribution_inr"]), reverse=True)

        return {
            "base_estimate_inr": round(base, 2),
            "xai_adjusted_cost_inr": round(total, 2),
            "top_drivers": drivers[:3]
        }