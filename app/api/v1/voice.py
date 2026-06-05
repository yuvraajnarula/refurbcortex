from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from app.core.security import verify_api_key
from app.services.voice_pipeline import MultilingualVoiceRouter
from app.utils.logger import app_logger

router = APIRouter()
voice_router_svc = MultilingualVoiceRouter(model_size="base")

@router.post("/transcribe")
async def transcribe_voice(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    api_key: str = Depends(verify_api_key)
):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Audio file required (wav/mp3)")
    
    try:
        audio_bytes = await file.read()
        result = await voice_router_svc.transcribe_and_route(audio_bytes, language)
        app_logger.info(f"🔊 Voice transcribed | Lang: {result['lang']} | Panels: {result['panels']}")
        
        return {"status": "success", "data": result}
    except Exception as e:
        app_logger.error(f"❌ Voice transcription failed: {e}")
        raise HTTPException(500, detail="Transcription engine unavailable")