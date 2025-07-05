# models/person_detector.py

import cv2
import numpy as np
from ultralytics import YOLO
from models.face_detector import FaceDetector

class PersonDetector:
    """
    Detects persons in images using YOLO and optionally verifies with face detection.
    Provides person mask extraction for background analysis.
    """

    def __init__(self, model_name='yolov8m-seg.pt', confidence_threshold=0.25, use_face_check=True):
        """
        Initializes the person detector with YOLO model and optional face verification.
        """
        self.model = YOLO(model_name)
        self.conf_threshold = confidence_threshold
        self.use_face_check = use_face_check
        self.face_detector = FaceDetector() if use_face_check else None
        self.person_class_id = 0  # YOLO class ID for 'person'

    def has_real_person(self, image_path):
        """
        Returns True if a real person is detected in the image.
        Optionally verifies by checking for at least one face.
        """
        image = self._load_image(image_path)
        if image is None:
            return False

        results = self._run_yolo(image)
        if not self._has_person_box(results):
            return False

        if not self.use_face_check:
            return True

        return self._has_face(image_path)

    def get_person_masks(self, image_path):
        """
        Returns a binary mask (same size as image) where person regions are marked as 255.
        Useful for background filtering and clutter analysis.
        """
        image = self._load_image(image_path)
        if image is None:
            return None

        results = self._run_yolo(image)
        if not self._has_masks(results):
            return None

        return self._combine_person_masks(results, image.shape)

    # === Private Helper Methods ===

    def _load_image(self, image_path):
        """Loads an image from disk. Returns None if unreadable."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not load image: {image_path}")
        return image

    def _run_yolo(self, image):
        """Runs YOLO model inference and returns the first result."""
        try:
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            return results[0] if results else None
        except Exception as e:
            print(f"[ERROR] YOLO inference failed: {e}")
            return None

    def _has_person_box(self, results):
        """Checks if at least one person box is detected."""
        if results is None or not hasattr(results, "boxes") or results.boxes is None:
            return False
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id == self.person_class_id:
                return True
        return False

    def _has_face(self, image_path):
        """Checks if at least one face is detected in the image."""
        if not self.face_detector:
            return False
        try:
            faces = self.face_detector.detect_faces(image_path)
            return len(faces) > 0
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            return False

    def _has_masks(self, results):
        """Checks if YOLO results contain segmentation masks."""
        return hasattr(results, "masks") and results.masks is not None

    def _combine_person_masks(self, results, image_shape):
        """
        Combines all person masks into a single binary mask.
        Returns mask with 255 for person regions, 0 elsewhere.
        """
        try:
            masks = results.masks.data.cpu().numpy()  # (N, H, W)
            boxes = results.boxes
            combined_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                if class_id == self.person_class_id:
                    mask = masks[i]
                    combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))

            return combined_mask * 255  # Convert to 0/255 binary mask
        except Exception as e:
            print(f"[ERROR] Failed to combine person masks: {e}")
            return None
