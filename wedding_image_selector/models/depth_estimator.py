

import torch
import cv2
import numpy as np

class DepthEstimator:
    """
    Estimates depth maps for images using the MiDaS DPT-Large model.
    """

    def __init__(self):
        """
        Initializes the depth estimator, loads the MiDaS model and its transforms.
        """
        self.device = self._get_device()
        self.model = self._load_midas_model()
        self.transform = self._load_midas_transform()

    def _get_device(self):
        """
        Returns the best available device (cuda or cpu).
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_midas_model(self):
        """
        Loads the MiDaS DPT-Large model from torch.hub.
        """
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        model.to(self.device)
        model.eval()
        return model

    def _load_midas_transform(self):
        """
        Loads the MiDaS transforms for preprocessing.
        """
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        return midas_transforms.dpt_transform

    def _preprocess_image(self, image):
        """
        Converts BGR image (OpenCV) to RGB and applies MiDaS transform.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image_rgb)["image"]
        return transformed.to(self.device).unsqueeze(0)

    def _postprocess_depth(self, prediction, original_size):
        """
        Resizes and normalizes the depth prediction.

        Args:
            prediction (torch.Tensor): Raw depth prediction.
            original_size (tuple): (height, width) of the original image.

        Returns:
            np.ndarray: Normalized depth map.
        """
        # Resize to original image size
        prediction_resized = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=original_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = prediction_resized.cpu().numpy()
        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        return depth

    def estimate_depth(self, image):
        """
        Estimates the normalized depth map for the given image.

        Args:
            image (np.ndarray): Input image (BGR format).

        Returns:
            np.ndarray: 2D array of normalized depth values in [0, 1].
        """
        original_size = image.shape[:2]
        input_tensor = self._preprocess_image(image)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            depth = self._postprocess_depth(prediction, original_size)

        return depth
