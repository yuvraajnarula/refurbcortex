import os
import numpy as np
import chromadb
from typing import Dict, List, Optional, Tuple
from app.config import settings
from app.utils.logger import app_logger

class MetacognitiveMonitor:
    def __init__(self):
        self.threshold = settings.CONFIDENCE_ROUTING_THRESH
        self.db_path = settings.CHROMA_DB_PATH
        os.makedirs(self.db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name="inspection_memory",
            metadata={"hnsw:space": "cosine"}
        )
        app_logger.info("Metacognitive Monitor initialized")

    def calibrate_confidence(self, raw_conf: float, damage_context: str) -> Tuple[float, float]:
        """
        Simulate ensemble/MC-Dropout variance. 
        Replace with actual model variance in production.
        """
        variance_proxy = 0.15 * (1 - raw_conf)
        calibrated = max(0.0, min(1.0, raw_conf - variance_proxy))
        return round(calibrated, 3), round(variance_proxy, 3)

    def query_memory(self, context: str) -> Optional[Dict]:
        """Lookup historical errors for similar panel+category+severity"""
        if self.collection.count() == 0:
            return None
        
        try:
            results = self.collection.query(
                query_texts=[context],
                n_results=3,
                include=["documents", "metadatas"]
            )
            if results["documents"] and results["documents"][0]:
                errors = [float(d) for d in results["documents"][0]]
                return {
                    "historical_error_median": round(float(np.median(errors)), 2),
                    "case_count": len(errors),
                    "safety_margin": round(float(np.median(errors) * 0.15), 2)
                }
        except Exception as e:
            app_logger.warning(f"Memory lookup failed: {e}")
        return None

    def route_decision(self, calibrated_conf: float, memory: Optional[Dict], base_cost: float) -> Dict:
        """Decide: CONFIDENT → proceed, UNCERTAIN → apply margin + flag human"""
        is_confident = calibrated_conf >= self.threshold
        adjusted_cost = base_cost
        
        if not is_confident and memory:
            adjusted_cost += memory["safety_margin"]
            action = "APPLY_MARGIN | FLAG_HUMAN_REVIEW"
        elif not is_confident:
            action = "REQUEST_BETTER_ANGLE | APPLY_DEFAULT_MARGIN"
            adjusted_cost *= 1.1
        else:
            action = "PROCEED_TO_SYSTEM_2"

        return {
            "status": "CONFIDENT" if is_confident else "UNCERTAIN",
            "confidence": calibrated_conf,
            "action": action,
            "adjusted_cost_inr": round(adjusted_cost, 2),
            "requires_human_review": not is_confident
        }

    def log_feedback(self, case_id: str, context: str, predicted: float, actual: float):
        """Store absolute error for future calibration"""
        error = abs(actual - predicted)
        self.collection.add(
            documents=[str(error)],
            metadatas=[{"case_id": case_id, "predicted": predicted, "actual": actual}],
            ids=[f"{case_id}_{context[:32]}"]
        )
        app_logger.info(f"Logged feedback: case={case_id} | error=₹{error}")