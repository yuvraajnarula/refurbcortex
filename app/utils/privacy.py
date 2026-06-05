# app/utils/privacy.py
import os
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from app.utils.logger import app_logger

UPLOAD_DIR = Path("./data/uploads")

def hash_pii(data: str) -> str:
    """Deterministic hashing for VIN/Names to allow search without storing raw PII."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def cleanup_old_uploads(retention_hours: int = 72):
    """Cron-job compatible function to delete raw images."""
    if not UPLOAD_DIR.exists():
        return

    now = datetime.now()
    deleted_count = 0

    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            # Check modification time
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            age_hours = (now - mtime).total_seconds() / 3600
            
            if age_hours > retention_hours:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    app_logger.error(f"❌ Failed to delete {file_path}: {e}")
    
    if deleted_count > 0:
        app_logger.info(f"🧹 Privacy Cleanup: Deleted {deleted_count} raw images > {retention_hours}h old")
