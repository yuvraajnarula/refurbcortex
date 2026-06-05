import sys
from loguru import logger
from app.config import settings

def setup_logger():
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>")
    return logger

app_logger = setup_logger()