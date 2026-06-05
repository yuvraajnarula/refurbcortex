import asyncio
from typing import Dict
from app.utils.logger import app_logger

class MetricsCollector:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.requests = self.inference_ms = self.errors = 0
        self.cost = 0.0
        self.err_window = []
        self.drift_flag = False
        self.drift_thresh = 0.15
        app_logger.info("📊 MetricsCollector initialized")

    async def record(self, inf_ms: float, is_err: bool, err_pct: float = 0.0, api_cost: float = 0.0):
        async with self.lock:
            self.requests += 1; self.inference_ms += inf_ms
            if is_err: self.errors += 1
            self.cost += api_cost
            self.err_window.append(err_pct)
            if len(self.err_window) > 50: self.err_window.pop(0)
            if len(self.err_window) >= 10:
                avg = sum(self.err_window[-10:]) / 10
                if avg > self.drift_thresh and not self.drift_flag:
                    self.drift_flag = True
                    app_logger.warning(f"🚨 CONCEPT DRIFT: Avg err {avg:.2%} > {self.drift_thresh:.2%}")
                elif avg < self.drift_thresh * 0.7:
                    self.drift_flag = False

    def snapshot(self) -> Dict:
        return {
            "requests": self.requests,
            "avg_latency_ms": round(self.inference_ms / max(1, self.requests), 1),
            "error_rate": round(self.errors / max(1, self.requests), 4),
            "unit_cost_inr": round(self.cost, 2),
            "drift_active": self.drift_flag,
            "ewma_error": round(sum(self.err_window[-10:])/max(1,len(self.err_window[-10:])), 4)
        }

metrics = MetricsCollector()