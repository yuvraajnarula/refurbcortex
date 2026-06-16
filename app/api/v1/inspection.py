# app/api/v1/inspection.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from app.core.system1_vision import System1Vision
from app.core.system2_agent import System2Agent
from app.core.xai_engine import XAIEngine
from app.api.v1.schema import InspectionResponse, RoutingDecision, AgentRecommendation
from app.utils.versioning import get_system_metadata
from app.core.security import verify_api_key
from app.services.web_agent_async import AsyncWebEnricher
from app.services.metrics_drift import metrics
from app.api.v1.webhooks import _status_db
from app.utils.logger import app_logger
import base64, cv2, time, uuid
from app.middleware.prometheus_metrics import record_confidence, record_ev_soh
from app.services.mlflow_registry import mlflow_reg

router = APIRouter()
vision_engine = System1Vision()
system2_engine = System2Agent()
xai_engine = XAIEngine()
web_agent = AsyncWebEnricher()

@router.post("/predict", response_model=InspectionResponse)
async def predict_damage(
    file: UploadFile = File(...), 
    vehicle_brand: str = Form(...), 
    vehicle_model: str = Form(...),
    manufacture_year: int = Form(...), 
    fuel_type: str = Form(...), 
    panel_affected: str = Form(...),
    damage_category: str = Form(...), 
    city: str = Form(...), 
    city_tier: str = Form(...),
    api_key: str = Depends(verify_api_key)
):
    start = time.time()
    insp_id = f"INS_{uuid.uuid4().hex[:8].upper()}"
    _status_db[insp_id] = {"inspection_id": insp_id, "status": "PROCESSING", "message": "Starting inference"}
    
    meta_dict = {
        "vehicle_brand": vehicle_brand, 
        "vehicle_model": vehicle_model, 
        "manufacture_year": manufacture_year,
        "fuel_type": fuel_type, 
        "panel_affected": panel_affected, 
        "damage_category": damage_category,
        "city": city, 
        "city_tier": city_tier, 
        "inspection_id": insp_id,
        "vehicle_age": 2024 - manufacture_year if manufacture_year < 2024 else 0,
        "odometer_km": 50000  # Extend via form param if needed
    }

    try:
        image_bytes = await file.read()
        # Updated signature: returns 4 values including routing
        heatmap_np, records, summary, avg_route = vision_engine.run_inference(image_bytes, meta_dict)

        rec, shap = None, None
        
        if avg_route["status"] == "CONFIDENT" and records:
            try:
                enrich_data = await web_agent.fetch(vehicle_brand, panel_affected, damage_category)
                rec_dicts = [r.model_dump() for r in records]
                
                rec = system2_engine.analyze_tradeoffs(rec_dicts, meta_dict)
                system2_engine.log_decision(insp_id, rec)
                
                shap = xai_engine.generate_breakdown(rec_dicts[0])
                
                # Prometheus metrics
                record_confidence(avg_route["confidence"])
                if fuel_type.lower() == "ev" and rec:
                    soh_val = float(rec.reasoning.split("EV SoH: ")[1].split("%")[0]) if "EV SoH: " in rec.reasoning else 85.0
                    record_ev_soh(vehicle_brand, vehicle_model, soh_val)

                # MLflow tracking
                inf_ms = round((time.time() - start) * 1000, 1)
                mlflow_reg.log_inference_run(
                    run_id=insp_id,
                    metrics={"confidence": avg_route["confidence"], "latency_ms": inf_ms},
                    params={"brand": vehicle_brand, "panel": panel_affected, "category": damage_category},
                    tags={"status": avg_route["status"], "version": "0.5.0-p2"}
                )
            except Exception as e:
                app_logger.warning(f"System2/XAI/Web fallback: {e}")
                rec = AgentRecommendation(
                    recommendation="PARTIAL", 
                    reasoning="Fallback: LLM/service unavailable. Rule-based threshold applied.", 
                    expected_net_margin_pct=10.0, 
                    turnover_risk="MED", 
                    repair_priority_items=[panel_affected]
                )

        # Encode heatmap
        _, buf = cv2.imencode('.jpg', heatmap_np)
        inf_ms = round((time.time() - start) * 1000, 1)
        
        # Metrics logging
        await metrics.record(inf_ms, is_error=False, api_cost=0.05)
        _status_db[insp_id] = {"inspection_id": insp_id, "status": "COMPLETED", "message": "Success", "ts": time.time()}

        return InspectionResponse(
            status="success" if avg_route["status"]=="CONFIDENT" else "uncertain",
            inspection_id=insp_id, 
            heatmap_b64=base64.b64encode(buf).decode(),
            damage_records=records, 
            summary=summary, 
            routing=RoutingDecision(**avg_route),
            recommendation=rec, 
            shap_breakdown=shap, 
            meta={**get_system_metadata(), "inference_time_ms": inf_ms}
        )
    except Exception as e:
        _status_db[insp_id] = {"inspection_id": insp_id, "status": "FAILED", "message": str(e), "ts": time.time()}
        await metrics.record(round((time.time()-start)*1000, 1), is_error=True)
        app_logger.error(f"❌ /predict failed | ID: {insp_id} | Error: {str(e)}")
        raise HTTPException(500, detail=f"Inspection pipeline failed: {str(e)}")

@router.get("/metrics")
async def get_metrics(api_key: str = Depends(verify_api_key)):
    return metrics.snapshot()