import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
import asyncio, os
from app.utils.versioning import get_system_metadata
from dotenv import load_dotenv

load_dotenv()

from app.middleware.idempotency import IdempotencyMiddleware
from app.middleware.prometheus_metrics import PrometheusMiddleware, metrics_endpoint, record_drift
from app.utils.privacy import cleanup_old_uploads
from app.services.offline_sync import OfflineSyncQueue
from app.services.metrics_drift import metrics
from app.utils.logger import app_logger

from app.api.v1.inspection import router as inspection_router
from app.api.v1.feedback import router as feedback_router
from app.api.v1.hitl import router as hitl_router
from app.api.v1.batch import router as batch_router
from app.api.v1.webhooks import router as webhook_router

from app.api.v1.insurance import router as insurance_router
from app.api.v1.voice import router as voice_router
from app.api.v1.twin import router as twin_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("🚀 RefurbCortex AI v0.5.0 Enterprise starting...")
    cleanup_old_uploads()
    app.state.sync_queue = OfflineSyncQueue()
    
    tasks = [
        asyncio.create_task(_periodic_cleanup()),
        asyncio.create_task(_periodic_sync()),
        asyncio.create_task(_drift_monitor())
    ]
    yield
    app_logger.info("🛑 Shutting down...")
    for t in tasks: t.cancel()

async def _periodic_cleanup():
    while True: await asyncio.sleep(3600); cleanup_old_uploads()

async def _periodic_sync():
    queue = app.state.sync_queue
    while True:
        await asyncio.sleep(60)
        try: await queue.flush_and_sync(lambda p: _mock_cloud_sync(p))
        except: pass

async def _drift_monitor():
    while True:
        record_drift(metrics.drift_flag)
        await asyncio.sleep(60)

async def _mock_cloud_sync(payload): pass  # Replace with real HTTP/aiohttp client

def create_app() -> FastAPI:
    app = FastAPI(
        title="RefurbCortex AI", 
        version="0.5.0", 
        lifespan=lifespan, 
        docs_url="/docs", 
        redoc_url="/redoc"
    )

    # 🔒 Middleware Stack
    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute", "10/second"])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(IdempotencyMiddleware)
    app.add_middleware(PrometheusMiddleware)  # Skips /metrics internally
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    # 🔌 Core Routers
    app.include_router(inspection_router, prefix="/api/v1", tags=["Inspection"])
    app.include_router(feedback_router, prefix="/api/v1", tags=["Feedback"])
    app.include_router(hitl_router, prefix="/api/v1", tags=["Human-in-the-Loop"])
    app.include_router(batch_router, prefix="/api/v1", tags=["Batch"])
    app.include_router(webhook_router, prefix="/api/v1/webhooks", tags=["Webhooks"])

    # 🧩 Conditional Moat Routers
    if os.getenv("ENABLE_INSURANCE", "false").lower() == "true":
        app.include_router(insurance_router, prefix="/api/v1/insurance", tags=["Insurance"])
    if os.getenv("ENABLE_VOICE", "false").lower() == "true":
        app.include_router(voice_router, prefix="/api/v1/voice", tags=["Voice"])
    if os.getenv("ENABLE_TWIN", "false").lower() == "true":
        app.include_router(twin_router, prefix="/api/v1/twin", tags=["Digital Twin"])

    # 📡 Observability & Health
    app.add_api_route("/metrics", metrics_endpoint, tags=["Observability"])
    @app.get("/health")
    async def health_check():
        try:
            import psutil, os, sqlite3
            # Track uptime
            if not hasattr(app.state, "start_time"): app.state.start_time = time.time()
            uptime = time.time() - app.state.start_time

            # System metrics
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent

            # Service dependency checks
            feedback_db = "connected" if os.path.exists("./data/feedback.db") else "pending"
            vector_db = "ready" if os.path.exists("./data/chroma_db") else "pending"
            queue_depth = 0
            if os.path.exists("./data/offline_queue.db"):
                with sqlite3.connect("./data/offline_queue.db") as conn:
                    queue_depth = conn.execute("SELECT COUNT(*) FROM pending_syncs").fetchone()[0]

            status = "healthy" if (cpu < 90 and mem < 95) else "degraded"

            return {
                "status": status,
                "version": "0.5.0-enterprise",
                "uptime_seconds": round(uptime, 1),
                "system_resources": {"cpu_pct": cpu, "memory_pct": mem, "disk_pct": disk},
                "services": {
                    "feedback_db": feedback_db,
                    "vector_memory": vector_db,
                    "offline_queue_depth": queue_depth,
                    "drift_active": metrics.drift_flag,
                    "total_requests": metrics.requests
                },
                "metadata": get_system_metadata()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "timestamp": time.time()}
    @app.get("/ready")
    def ready(): 
        return {"status": "ready"}

    return app

app = create_app()