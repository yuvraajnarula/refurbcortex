from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio, time, uuid, base64
from app.core.security import verify_api_key
from app.core.system1_vision import System1Vision
from app.core.system2_agent import System2Agent
from app.core.xai_engine import XAIEngine
from app.services.metrics_drift import metrics
from app.utils.versioning import get_system_metadata
from app.utils.logger import app_logger

router = APIRouter()
vision_engine = System1Vision()
system2_engine = System2Agent()
xai_engine = XAIEngine()

class BatchItem(BaseModel):
    image_b64: str
    vehicle_brand: str
    vehicle_model: str
    manufacture_year: int
    fuel_type: str
    panel_affected: str
    damage_category: str
    city: str
    city_tier: str

class BatchRequest(BaseModel):
    items: List[BatchItem] = Field(..., min_items=1, max_items=20)
    priority: str = "standard"  # "standard" | "high"

@router.post("/batch_predict")
async def batch_predict(req: BatchRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()
    batch_id = f"BATCH_{uuid.uuid4().hex[:8].upper()}"
    max_concurrency = 5 if req.priority == "high" else 3
    semaphore = asyncio.Semaphore(max_concurrency)
    results = []
    errors = []

    async def process_item(item: BatchItem, idx: int):
        async with semaphore:
            item_id = f"{batch_id}_ITEM_{idx+1}"
            try:
                img_bytes = base64.b64decode(item.image_b64)
                meta = item.model_dump()
                meta["inspection_id"] = item_id

                heatmap_np, records, summary, avg_route = vision_engine.run_inference(img_bytes, meta)
                rec, shap = None, None

                if avg_route["status"] == "CONFIDENT" and records:
                    try:
                        rec = system2_engine.analyze_tradeoffs([r.model_dump() for r in records], meta)
                        system2_engine.log_decision(item_id, rec)
                        shap = xai_engine.generate_breakdown(records[0].model_dump())
                    except Exception as e:
                        app_logger.warning(f"⚠️ System2/XAI fallback for {item_id}: {e}")

                results.append({
                    "item_id": item_id,
                    "status": "success",
                    "routing": avg_route,
                    "recommendation": rec.model_dump() if rec else None,
                    "shap_breakdown": shap,
                    "summary": summary
                })
            except Exception as e:
                app_logger.error(f"❌ Batch item {item_id} failed: {e}")
                errors.append({"item_id": item_id, "status": "error", "detail": str(e)})

    tasks = [process_item(item, i) for i, item in enumerate(req.items)]
    await asyncio.gather(*tasks)

    proc_time_ms = round((time.time() - start_time) * 1000, 1)
    await metrics.record(proc_time_ms, is_error=len(errors) > 0)

    return {
        "batch_id": batch_id,
        "total_items": len(req.items),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
        "processing_time_ms": proc_time_ms,
        "meta": get_system_metadata()
    }