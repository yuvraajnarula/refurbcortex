# app/api/v1/twin.py
from fastapi import APIRouter, Query, HTTPException, Depends
from app.core.security import verify_api_key
from app.services.digital_twin import DigitalTwinForecaster
from app.utils.logger import app_logger

router = APIRouter()
twin_svc = DigitalTwinForecaster()

@router.get("/project")
async def project_wear(
    inspection_id: str = Query(...),
    forward_km: int = Query(20000, ge=1000, le=100000),
    api_key: str = Depends(verify_api_key)
):
    # Mock current state (replace with DB lookup in prod)
    current_state = {
        "battery_soh": 88.0,
        "paint_integrity": 72.0,
        "suspension_life": 65.0,
        "brake_pad_thickness": 8.5
    }
    
    try:
        projection = twin_svc.simulate(current_state, forward_km)
        app_logger.info(f"🔮 Twin projection: {inspection_id} | {forward_km}km")
        
        return {
            "status": "success",
            "inspection_id": inspection_id,
            "projection_km": forward_km,
            "components": projection
        }
    except Exception as e:
        app_logger.error(f"❌ Twin projection failed: {e}")
        raise HTTPException(500, detail="Degradation model unavailable")