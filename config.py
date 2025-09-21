# Configuration for connecting services
ML_SERVICE_CONFIG = {
    "huggingface": {
        "url": "https://your-username-sih-isro-ml-service.hf.space/api/predict",
        "timeout": 30,
        "retry_attempts": 3
    },
    "local": {
        "url": "http://localhost:7860/api/predict",
        "timeout": 10,
        "retry_attempts": 1
    }
}

# Environment-based selection
import os
ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')
ML_SERVICE_URL = ML_SERVICE_CONFIG['huggingface']['url'] if ENVIRONMENT == 'production' else ML_SERVICE_CONFIG['local']['url']
