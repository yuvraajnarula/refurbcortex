from pydantic_settings import BaseSettings
from typing import Optional
import os 
class Settings(BaseSettings):
    APP_ENV: str = "development"
    YOLO_MODEL_PATH: str = "./models/yolov8s.pt"
    REPAIR_COST_PATH: str = "./data/repair_costs.csv"
    CONFIDENCE_THRESH: float = 0.45
    HEATMAP_OPACITY: float = 0.35
    LOG_LEVEL: str = "INFO"
    API_KEY: str = os.getenv("API_KEY", "")  

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()