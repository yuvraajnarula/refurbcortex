from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.config import settings
from app.utils.logger import app_logger

security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if settings.API_KEY and credentials.credentials != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired API key")
    return credentials.credentials