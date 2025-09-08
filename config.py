# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Model settings
    embed_model: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    threshold: float = float(os.getenv("THRESHOLD", "0.68"))
    top_k: int = int(os.getenv("TOP_K", "25"))
    
    # File limits
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
    
    # UI settings
    port: int = int(os.getenv("PORT", "8501"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Security
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT", "100"))
    rate_limit_window_minutes: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

config = Config()
