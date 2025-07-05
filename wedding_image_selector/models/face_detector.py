

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from utils.image_utils import compress_image
import config

class FaceDetector:
    def __init__(self, det_size=(640, 640)):
        """
        Initialize the FaceDetector with model and configuration.
        """
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.sharpness_threshold = config.SHARPNESS_THRESHOLD
        self.crop_margin_ratio = getattr(config, 'CROPPED_FACE_MARGIN', 0.05)

    def detect_faces(self, image_path):
        """
        Main method to detect faces and their attributes in an image.
        Returns a list of dicts with face info.
        """
        image = self._load_and_preprocess_image(image_path)
        if image is None:
            return []

        if self._is_blurry(image):
            print(f"[INFO] Blurry image skipped: {image_path}")
            return []

        faces = self.app.get(image)
        if not faces:
            return []

        return self._process_faces(image, faces)

    def _load_and_preprocess_image(self, image_path):
        """
        Loads and compresses the image.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return None
        image = compress_image(image)
        if image is None:
            print(f"[ERROR] Invalid image after compression: {image_path}")
            return None
        return image

    def _is_blurry(self, image):
        """
        Checks if the image is blurry based on Laplacian variance.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return sharpness < self.sharpness_threshold

    def _process_faces(self, image, faces):
        """
        Processes detected faces, extracting attributes and filtering.
        """
        height, width = image.shape[:2]
        margin_x = self.crop_margin_ratio * width
        margin_y = self.crop_margin_ratio * height
        results = []

        for face in faces:
            bbox = self._clamp_bbox(face.bbox.astype(int), width, height)
            face_crop = self._crop_face(image, bbox)
            if face_crop.size == 0:
                continue

            is_cropped = self._is_face_cropped(bbox, width, height, margin_x, margin_y)
            has_sunglasses = self._detect_sunglasses(face, face_crop)

            results.append({
                "bbox": bbox,
                "landmarks": face.kps.tolist() if hasattr(face, "kps") else [],
                "is_cropped": is_cropped,
                "has_sunglasses": has_sunglasses
            })
        return results

    def _clamp_bbox(self, bbox, width, height):
        """
        Ensures bounding box is within image bounds.
        """
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        return [x1, y1, x2, y2]

    def _crop_face(self, image, bbox):
        """
        Crops the face region from the image.
        """
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]

    def _is_face_cropped(self, bbox, width, height, margin_x, margin_y):
        """
        Checks if the face is close to the image borders.
        """
        x1, y1, x2, y2 = bbox
        return (
            x1 <= margin_x or y1 <= margin_y or
            x2 >= (width - margin_x) or y2 >= (height - margin_y)
        )

    def _detect_sunglasses(self, face, face_crop):
        """
        Detects sunglasses by analyzing the darkness of the eye region.
        """
        try:
            kps = getattr(face, "kps", None)
            if kps is not None and len(kps) >= 5:
                eye_region = self.extract_eye_region(face_crop, kps)
                if eye_region.size > 0:
                    hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
                    v = hsv[:, :, 2]
                    dark_ratio = np.sum(v < 50) / (v.size + 1e-6)
                    return dark_ratio > 0.5
        except Exception:
            pass
        return False

    def extract_eye_region(self, face_crop, kps):
        """
        Extracts the region between both eyes from the face crop.
        """
        try:
            left_eye = kps[0]
            right_eye = kps[3]
            mid = (left_eye + right_eye) / 2
            eye_w = int(np.linalg.norm(right_eye - left_eye))
            eye_h = int(eye_w * 0.6)

            cx, cy = int(mid[0]), int(mid[1])
            x1 = max(0, cx - eye_w // 2)
            y1 = max(0, cy - eye_h // 2)
            x2 = min(face_crop.shape[1], cx + eye_w // 2)
            y2 = min(face_crop.shape[0], cy + eye_h // 2)

            return face_crop[y1:y2, x1:x2]
        except Exception:
            return np.zeros((1, 1, 3), dtype=np.uint8)

