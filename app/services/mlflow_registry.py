# app/services/mlflow_registry.py
import os
import mlflow
from typing import Dict, Optional
from app.utils.logger import app_logger

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./data/mlflow.db")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("RefurbCortex_AI")

class MLflowRegistry:
    def __init__(self):
        app_logger.info(f"🔮 MLflow initialized: {MLFLOW_URI}")

    def log_inference_run(self, run_id: str, metrics: Dict, params: Dict, tags: Dict = None):
        try:
            with mlflow.start_run(run_name=run_id):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                if tags:
                    for k, v in tags.items():
                        mlflow.set_tag(k, v)
        except Exception as e:
            app_logger.warning(f"⚠️ MLflow log failed (fallback active): {e}")

    def get_production_model_version(self, model_name: str) -> str:
        """Returns production version or deterministic fallback"""
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            prod = [v for v in versions if v.current_stage == "Production"]
            return prod[0].version if prod else "0.1.0-fallback"
        except Exception:
            return "0.1.0-fallback"

mlflow_reg = MLflowRegistry()