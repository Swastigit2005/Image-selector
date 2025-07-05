

import numpy as np
import cv2
import insightface

class CompositionScorer:
    """
    Computes various composition-related scores for an image, such as
    center alignment, rule of thirds, golden spiral, symmetry, and color harmony.
    """

    def __init__(self, img_shape):
        """
        Initializes the scorer with image dimensions and key composition points.
        Args:
            img_shape (tuple): Shape of the image (height, width, ...).
        """
        self.height, self.width = img_shape[:2]
        self.center = np.array([self.width / 2, self.height / 2])
        self.thirds_points = self._compute_rule_of_thirds_points()
        # --- initialize insightface detector ---
        self.face_detector = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.face_detector.prepare(ctx_id=0, det_size=(max(self.width, 640), max(self.height, 640)))

    def detect_face_centers(self, image, min_area=0.01, min_landmarks=1, edge_margin=0.02):
        """
        Detects faces in the image and returns their center coordinates.
        Args:
            image (np.ndarray): Input image (BGR).
            min_area (float): Minimum area for a face to be considered valid (fraction of image area).
            min_landmarks (int): Minimum number of landmarks required.
            edge_margin (float): Margin (fraction of image size) to avoid faces too close to the edge.
        Returns:
            list of (x, y): List of valid face center coordinates.
        """
        faces = self.face_detector.get(image)
        print(f"[DEBUG] Detected faces: {len(faces)}")  # Debug log
        centers = []
        h, w = image.shape[:2]
        margin_x = edge_margin * w
        margin_y = edge_margin * h
        for face in faces:
            box = face.bbox  # [x1, y1, x2, y2]
            area = (box[2] - box[0]) * (box[3] - box[1])
            landmarks = getattr(face, "landmark", None)
            num_landmarks = len(landmarks) if landmarks is not None else 0
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            # Relaxed filtering: only filter out faces that are extremely small, have no landmarks, or are almost out of frame
            if (
                area >= min_area * w * h and
                num_landmarks >= min_landmarks and
                margin_x <= x_center <= (w - margin_x) and
                margin_y <= y_center <= (h - margin_y)
            ):
                centers.append((x_center, y_center))
        print(f"[DEBUG] Valid faces after filtering: {len(centers)}")  # Debug log
        return centers

    def _compute_rule_of_thirds_points(self):
        """Returns the four intersection points for the rule of thirds grid."""
        return [
            (self.width / 3, self.height / 3),
            (2 * self.width / 3, self.height / 3),
            (self.width / 3, 2 * self.height / 3),
            (2 * self.width / 3, 2 * self.height / 3)
        ]

    def face_center_score(self, face_centers):
        """
        Scores how close face centers are to the image center.
        Args:
            face_centers (list of (x, y)): List of face center coordinates.
        Returns:
            float: Score in [0.0, 1.0].
        """
        if not face_centers:
            return 0.0
        max_dist = np.linalg.norm([self.width, self.height]) / 2.0
        scores = [
            1.0 - np.linalg.norm(np.array(fc) - self.center) / max_dist
            for fc in face_centers
        ]
        return round(float(np.clip(np.mean(scores), 0.0, 1.0)), 2)

    def rule_of_thirds_score(self, face_centers):
        """
        Scores how close face centers are to rule of thirds intersections.
        Args:
            face_centers (list of (x, y)): List of face center coordinates.
        Returns:
            float: Score in [0.0, 1.0].
        """
        if not face_centers:
            return 0.0
        max_dist = np.linalg.norm([self.width, self.height])
        scores = [
            1.0 - (min(
                float(np.linalg.norm(np.array(fc) - np.array(tp)))
                for tp in self.thirds_points
            ) / max_dist)
            for fc in face_centers
        ]
        return round(float(np.clip(np.mean(scores), 0.0, 1.0)), 2)

    def golden_spiral_score(self, face_centers):
        """
        Scores how well face centers align with golden spiral origins.
        Args:
            face_centers (list of (x, y)): List of face center coordinates.
        Returns:
            float: Score in [0.0, 1.0].
        """
        if not face_centers:
            return 0.0

        spiral_origins = self._get_golden_spiral_origins()
        max_dist = np.sqrt(self.width**2 + self.height**2) / 2.0

        def score_point(p, target):
            dist = np.linalg.norm(np.array(p) - np.array(target))
            return max(0.0, 1.0 - dist / max_dist)

        scores = [
            max(score_point(face, origin) for origin in spiral_origins)
            for face in face_centers
        ]
        return round(float(np.clip(np.mean(scores), 0.0, 1.0)), 3)

    def _get_golden_spiral_origins(self):
        """Returns four classical golden spiral origins for the image."""
        w, h = self.width, self.height
        return [
            (0.618 * w, 0.382 * h),  # Top-left
            (0.382 * w, 0.382 * h),  # Top-right
            (0.382 * w, 0.618 * h),  # Bottom-right
            (0.618 * w, 0.618 * h),  # Bottom-left
        ]

    def symmetry_score(self, face_centers):
        """
        Scores horizontal symmetry of face centers about the vertical axis.
        Args:
            face_centers (list of (x, y)): List of face center coordinates.
        Returns:
            float: Score in [0.0, 1.0].
        """
        if not face_centers:
            return 0.0
        scores = [
            1.0 - abs(fc[0] - self.center[0]) / (self.width / 2.0)
            for fc in face_centers
        ]
        return round(float(np.clip(np.mean(scores), 0.0, 1.0)), 2)

    def color_harmony_score(self, image):
        """
        Scores color harmony based on saturation and brightness uniformity.
        Args:
            image (np.ndarray): Input image (BGR).
        Returns:
            float: Score in [0.0, 1.0].
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_std = np.std(hsv[:, :, 1]) / 255.0  # Saturation std-dev
        v_std = np.std(hsv[:, :, 2]) / 255.0  # Brightness std-dev
        harmony = 1.0 - ((s_std + v_std) / 2.0)
        return round(float(np.clip(harmony, 0.0, 1.0)), 2)

    def get_face_centers_for_scoring(self, image):
        """
        Detects and returns valid face centers for scoring.
        Args:
            image (np.ndarray): Input image (BGR).
        Returns:
            list of (x, y): List of valid face center coordinates.
        """
        return self.detect_face_centers(image)

# Example usage in your pipeline:
# image = cv2.imread(image_path)
# scorer = CompositionScorer(image.shape)
# face_centers = scorer.get_face_centers_for_scoring(image)
# center_score = scorer.face_center_score(face_centers)
# thirds_score = scorer.rule_of_thirds_score(face_centers)
# ...etc...
