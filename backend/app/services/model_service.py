import threading
from typing import List, Union
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from app.core.config import get_settings
from app.core.logging import setup_logging

logger = setup_logging()
settings = get_settings()


class ModelService:
    """Singleton SigLIP-2 model service. Loaded once and reused."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SigLIP-2 model '{settings.MODEL_NAME}' on {self.device}")
        self.processor = AutoProcessor.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModel.from_pretrained(settings.MODEL_NAME).to(self.device)
        self.model.eval()
        self._initialized = True
        logger.info("SigLIP-2 model loaded successfully")

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """Encode a batch of PIL Images into L2-normalized embeddings."""
        if not images:
            return np.array([])
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb.cpu().numpy()

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text query/queries into L2-normalized embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb.cpu().numpy()


# Global singleton instance (lazy init on first use)
_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
