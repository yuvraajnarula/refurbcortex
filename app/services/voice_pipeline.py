import os
from faster_whisper import WhisperModel
from langdetect import detect
from app.utils.logger import app_logger

REGIONAL_MAP = {
    "bumper": "front_bumper", "bonnet": "hood", "dabba": "door",
    "kaanch": "windshield", "paint_utra": "scratch", "gaddha": "dent"
}

class MultilingualVoiceRouter:
    def __init__(self, model_size="base"):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        app_logger.info("Local Whisper loaded (CPU int8)")

    async def transcribe_and_route(self, audio_bytes: bytes, target_lang: str = "auto") -> dict:
        segments, _ = self.model.transcribe(audio_bytes, beam_size=5, language=target_lang if target_lang != "auto" else None)
        text = " ".join([seg.text for seg in segments]).strip()
        lang = detect(text) if target_lang == "auto" else target_lang
        
        # Normalize regional terms → standard panels
        normalized = text.lower()
        for regional, standard in REGIONAL_MAP.items():
            if regional in normalized: normalized = normalized.replace(regional, standard)
            
        return {"text": text, "lang": lang, "normalized": normalized, "panels": [p for p in REGIONAL_MAP.values() if p in normalized]}