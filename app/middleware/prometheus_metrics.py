# app/middleware/prometheus_metrics.py
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse
import time, threading

# Thread-safe metrics
_REQUEST_COUNT = Counter("refurb_request_total", "Total requests", ["method", "endpoint", "status"])
_REQUEST_LATENCY = Histogram("refurb_request_latency_seconds", "Request latency", ["endpoint"])
_CONFIDENCE_DIST = Histogram("refurb_model_confidence", "Model confidence distribution", buckets=[0.1*i for i in range(11)])
_EV_SOH_GAUGE = Gauge("refurb_ev_soh_pred_pct", "Predicted EV State of Health %", ["vehicle_brand", "vehicle_model"])
_DRIFT_ACTIVE = Gauge("refurb_drift_active", "Concept drift flag (1=active)")

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            duration = time.time() - start
            endpoint = request.url.path.split("/")[3] if len(request.url.path.split("/")) > 3 else "unknown"
            _REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status=status).inc()
            _REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        return response

def record_confidence(conf: float):
    _CONFIDENCE_DIST.observe(conf)

def record_ev_soh(brand: str, model: str, soh: float):
    _EV_SOH_GAUGE.labels(vehicle_brand=brand, vehicle_model=model).set(soh)

def record_drift(flag: bool):
    _DRIFT_ACTIVE.set(1 if flag else 0)

def metrics_endpoint():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)