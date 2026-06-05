import os
import sqlite3
from typing import Dict
from app.utils.logger import app_logger
from app.core.metacognitive import MetacognitiveMonitor

class FeedbackLoop:
    def __init__(self, db_path: str = "./data/feedback.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.monitor = MetacognitiveMonitor()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_logs (
                    inspection_id TEXT PRIMARY KEY,
                    predicted_cost REAL,
                    actual_cost REAL,
                    absolute_error REAL,
                    error_rate REAL,
                    human_override BOOLEAN DEFAULT 0,
                    notes TEXT,
                    logged_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def record(self, inspection_id: str, predicted_cost: float, actual_cost: float, human_override: bool = False, notes: str = "") -> Dict:
        abs_error = abs(actual_cost - predicted_cost)
        error_rate = abs_error / max(predicted_cost, 1.0)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feedback_logs 
                (inspection_id, predicted_cost, actual_cost, absolute_error, error_rate, human_override, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (inspection_id, predicted_cost, actual_cost, abs_error, error_rate, human_override, notes))

        self.monitor.log_feedback(
            case_id=inspection_id,
            context=f"cost_error_{inspection_id}",
            predicted=predicted_cost,
            actual=actual_cost
        )

        app_logger.info(f"Feedback Logged | {inspection_id} | Pred: Rs{predicted_cost} | Act: Rs{actual_cost} | Err: {error_rate:.2%}")

        return {
            "inspection_id": inspection_id,
            "absolute_error_inr": round(abs_error, 2),
            "error_rate_pct": round(error_rate * 100, 2),
            "memory_updated": True
        }