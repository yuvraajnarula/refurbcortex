from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_ENV: str = "development"
    YOLO_MODEL_PATH: str = "./models/yolov8s.pt"
    REPAIR_COST_PATH: str = "./data/india_vehicle_refurbishment_dataset.csv"  
    CONFIDENCE_THRESH: float = 0.45
    HEATMAP_OPACITY: float = 0.35
    LOG_LEVEL: str = "INFO"
    API_KEY: str = ""

    # Feature flags
    enable_airgapped: bool = False
    enable_insurance: bool = False
    enable_multilingual_voice: bool = False
    enable_digital_twin: bool = False

    # System 2 Agent (Groq)
    SYSTEM2_MODEL_NAME: str = "llama3-8b-8192"
    groq_api_key: str = ""

    # New missing fields
    CONFIDENCE_ROUTING_THRESH: float = 0.8
    CHROMA_DB_PATH: str = "./data/chroma_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=False,
    )

settings = Settings()