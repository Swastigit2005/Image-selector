

import os
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel


class AestheticHead(nn.Module):
    """
    Neural network head for predicting aesthetic score from CLIP image features.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Linear(1024, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScorer:
    """
    Computes an aesthetic score for an image using CLIP and a custom regression head.
    """
    def __init__(self, weights_path=None):
        self.device = self._get_device()
        self.model_name = "openai/clip-vit-large-patch14"
        self.clip = self._load_clip_model()
        self.processor = self._load_clip_processor()
        self.head = self._load_aesthetic_head(weights_path)
        self.head.eval()

    def _get_device(self):
        """Returns the best available device (cuda or cpu)."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _load_clip_model(self):
        """Loads the CLIP model."""
        return CLIPModel.from_pretrained(self.model_name).to(self.device)

    def _load_clip_processor(self):
        """Loads the CLIP processor."""
        return CLIPProcessor.from_pretrained(self.model_name)

    def _load_aesthetic_head(self, weights_path):
        """Initializes and loads weights for the aesthetic head."""
        head = AestheticHead().to(self.device)
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__), "ava_logos_linearMSE.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"[ERROR] Aesthetic weights not found at: {weights_path}")
        head.load_state_dict(torch.load(weights_path, map_location=self.device))
        return head

    def score(self, image_path: str) -> float:
        """
        Computes the aesthetic score for a given image path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            float: Aesthetic score in [0.0, 1.0]. Returns 0.0 on error.
        """
        image = self._load_image(image_path)
        if image is None:
            return 0.0

        try:
            features = self._extract_clip_features(image)
            score = self._predict_aesthetic_score(features)
            return self._sanitize_score(score)
        except Exception as e:
            print(f"[AestheticScorer] Error processing {image_path}: {e}")
            return 0.0

    def _load_image(self, image_path):
        """Loads and converts an image to RGB. Returns None on failure."""
        try:
            return Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError):
            return None

    def _extract_clip_features(self, image):
        """Extracts normalized CLIP image features."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.clip.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features

    def _predict_aesthetic_score(self, features):
        """Predicts the aesthetic score using the regression head."""
        with torch.no_grad():
            score_tensor = self.head(features)
            return score_tensor.squeeze().detach().cpu().item()

    def _sanitize_score(self, score):
        """Clamps and rounds the score to [0.0, 1.0] with 2 decimals."""
        return round(min(max(score, 0.0), 1.0), 2)
