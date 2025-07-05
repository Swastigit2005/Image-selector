

import cv2
import numpy as np
from sklearn.cluster import KMeans

class BackgroundClutterScorer:
    """
    Scores the background clutter of an image using edge density and color complexity.
    Optionally focuses on background regions using a person mask.
    """

    def __init__(self, clutter_threshold=0.6):
        """
        Args:
            clutter_threshold (float): Threshold above which background is considered cluttered.
        """
        self.clutter_threshold = clutter_threshold  # 0 = clean, 1 = chaotic

    def compute_clutter_score(self, image, person_mask=None):
        """
        Computes a clutter score for the image.
        Args:
            image (np.ndarray): Input image (BGR).
            person_mask (np.ndarray or None): Optional mask (1=person, 0=background).
        Returns:
            float: Clutter score in [0.0, 1.0].
        """
        small, gray = self._preprocess_image(image)
        background_mask = self._get_background_mask(gray, person_mask)
        edge_density = self._compute_edge_density(gray, background_mask)
        color_complexity = self._compute_color_complexity(small, background_mask)
        clutter_score = self._combine_scores(edge_density, color_complexity)
        return clutter_score

    def is_background_clean(self, image, person_mask=None):
        """
        Returns True if the background is considered clean.
        """
        return self.compute_clutter_score(image, person_mask) < self.clutter_threshold

    # === Helper methods ===

    def _preprocess_image(self, image):
        """
        Resizes image to 256x256 and converts to grayscale.
        Returns:
            small (np.ndarray): Resized BGR image.
            gray (np.ndarray): Grayscale image.
        """
        small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return small, gray

    def _get_background_mask(self, gray, person_mask):
        """
        Returns a mask where background pixels are 1, others 0.
        Args:
            gray (np.ndarray): Grayscale image for shape reference.
            person_mask (np.ndarray or None): Optional mask.
        Returns:
            np.ndarray: Background mask.
        """
        if person_mask is not None:
            person_mask = cv2.resize(person_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            return (person_mask == 0).astype(np.uint8)
        return np.ones_like(gray, dtype=np.uint8)

    def _compute_edge_density(self, gray, background_mask):
        """
        Computes edge density in the background.
        Args:
            gray (np.ndarray): Grayscale image.
            background_mask (np.ndarray): Background mask.
        Returns:
            float: Edge density in [0.0, 1.0].
        """
        edges = cv2.Canny(gray, 100, 200)
        edge_pixels = np.sum((edges > 0) & (background_mask > 0))
        total_bg_pixels = np.sum(background_mask)
        if total_bg_pixels == 0:
            return 0.0
        return edge_pixels / total_bg_pixels

    def _compute_color_complexity(self, small, background_mask):
        """
        Computes color complexity (entropy) in the background.
        Args:
            small (np.ndarray): Resized BGR image.
            background_mask (np.ndarray): Background mask.
        Returns:
            float: Color complexity in [0.0, 1.0].
        """
        pixels = small[background_mask > 0].reshape(-1, 3)
        if len(pixels) < 5:
            return 1.0  # Not enough pixels to cluster
        try:
            kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
            labels = kmeans.fit_predict(pixels)
            _, counts = np.unique(labels, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
            return entropy / np.log2(5)
        except Exception:
            return 1.0

    def _combine_scores(self, edge_density, color_complexity):
        """
        Combines edge density and color complexity into a single clutter score.
        Returns:
            float: Clutter score in [0.0, 1.0], rounded to 3 decimals.
        """
        clutter_score = 0.5 * edge_density + 0.5 * color_complexity
        return round(min(max(clutter_score, 0.0), 1.0), 3)
