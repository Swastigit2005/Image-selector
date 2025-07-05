import insightface
import numpy as np
import cv2

class InsightFaceDetector:
    def __init__(self, det_size=(640, 640)):
        self.model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=det_size)

    def detect_faces(self, image):
        """
        Detect faces and return face objects from insightface.
        Args:
            image (np.ndarray): BGR image.
        Returns:
            List of face objects.
        """
        faces = self.model.get(image)
        print(f"[DEBUG] Detected faces: {len(faces)}")  # Debug log
        return faces

    def detect_face_centers(self, image, min_area=0, min_landmarks=0):
        """
        Detect faces and return their center coordinates, optionally filtering by area or landmarks.
        Args:
            image (np.ndarray): BGR image.
            min_area (float): Minimum area for a face to be considered valid.
            min_landmarks (int): Minimum number of landmarks required.
        Returns:
            List of (x, y) tuples for valid face centers.
        """
        faces = self.detect_faces(image)
        centers = []
        h, w = image.shape[:2]
        for face in faces:
            box = face.bbox  # [x1, y1, x2, y2]
            area = (box[2] - box[0]) * (box[3] - box[1])
            landmarks = getattr(face, "landmark", None)
            num_landmarks = len(landmarks) if landmarks is not None else 0
            # Filter by area and landmarks if specified
            if (min_area == 0 or area >= min_area * w * h) and (min_landmarks == 0 or num_landmarks >= min_landmarks):
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                centers.append((x_center, y_center))
        print(f"[DEBUG] Valid faces after filtering: {len(centers)}")  # Debug log
        return centers
