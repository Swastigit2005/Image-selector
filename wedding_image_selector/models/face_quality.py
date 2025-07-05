

import cv2
import numpy as np
import config

class FaceQualityScorer:
    """
    Evaluates the quality of detected faces in an image based on sharpness, brightness,
    pose, landmark presence, eye openness, occlusion, and other configurable criteria.
    """

    def __init__(self):
        self.sharpness_thresh = config.SHARPNESS_THRESHOLD
        self.brightness_thresh = config.BRIGHTNESS_THRESHOLD
        self.min_eye_open_ratio = config.MIN_EYE_OPEN_RATIO
        self.max_yaw = config.MAX_POSE_YAW
        self.max_pitch = config.MAX_POSE_PITCH
        self.min_landmarks = config.MIN_LANDMARKS_REQUIRED
        self.disallowed_emotions = {"angry", "disgust", "fear", "sad"}

    def is_face_good(self, face, image, image_type="group"):
        """
        Main entry: Returns True if the face meets all quality and clarity criteria.
        """
        x1, y1, x2, y2 = face["bbox"]
        crop = image[y1:y2, x1:x2]
        if not self._is_valid_crop(crop):
            return False

        if face.get("is_cropped", False):
            return False

        if not self._passes_age_filter(face, image_type):
            return False

        if not self._passes_pose_filter(face):
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if not self._passes_sharpness(gray):
            return False

        if not self._passes_brightness(gray):
            return False

        kps = face.get("landmarks", [])
        if not self._has_valid_landmarks(kps):
            return False

        if not face.get("has_sunglasses", False):
            if not self._passes_eye_openness_and_occlusion(kps, gray):
                return False

        # Optional: Uncomment to reject based on emotion for non-group images
        # if face.get("emotion", "").lower() in self.disallowed_emotions and image_type != "group":
        #     return False

        return True

    # === Modular Helper Methods ===

    def _is_valid_crop(self, crop):
        """Checks if the face crop is non-empty."""
        return crop is not None and crop.size > 0

    def _passes_age_filter(self, face, image_type):
        """
        Rejects children (age <= 15) except in group shots.
        """
        age = face.get("age")
        if age is not None:
            try:
                if age <= 15 and image_type != "group":
                    return False
            except Exception:
                pass
        return True

    def _passes_pose_filter(self, face):
        """
        Checks if face yaw and pitch are within allowed limits.
        """
        yaw = face.get("yaw")
        pitch = face.get("pitch")
        if yaw is not None and abs(yaw) > self.max_yaw:
            return False
        if pitch is not None and abs(pitch) > self.max_pitch:
            return False
        return True

    def _passes_sharpness(self, gray_crop):
        """
        Checks if the face crop is sharp enough.
        """
        sharpness = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
        return sharpness >= self.sharpness_thresh

    def _passes_brightness(self, gray_crop):
        """
        Checks if the face crop has acceptable brightness.
        """
        brightness = np.mean(gray_crop)
        return self.brightness_thresh[0] <= brightness <= self.brightness_thresh[1]

    def _has_valid_landmarks(self, kps):
        """
        Checks if the face has enough landmarks.
        """
        return kps is not None and len(kps) >= self.min_landmarks

    def _passes_eye_openness_and_occlusion(self, kps, gray_crop):
        """
        Checks eye openness, per-eye sharpness, and occlusion/symmetry.
        """
        try:
            # Eye openness
            eye_height = abs(kps[1][1] - kps[0][1])
            eye_width = np.linalg.norm(np.array(kps[1]) - np.array(kps[0])) + 1e-6
            eye_ratio = eye_height / eye_width
            if eye_ratio < self.min_eye_open_ratio:
                return False

            # Per-eye sharpness
            if not self._are_eyes_sharp(kps, gray_crop):
                return False

            # Occlusion and symmetry checks
            if not self._passes_occlusion_symmetry(kps):
                return False

            return True
        except Exception:
            return False

    def _are_eyes_sharp(self, kps, gray_crop):
        """
        Checks if both eyes are sharp enough.
        """
        eye_crop_size = 20
        for (ex, ey) in [kps[0], kps[1]]:
            ex, ey = int(ex), int(ey)
            eye_crop = gray_crop[max(0, ey - eye_crop_size):ey + eye_crop_size,
                                 max(0, ex - eye_crop_size):ex + eye_crop_size]
            if eye_crop.size == 0 or cv2.Laplacian(eye_crop, cv2.CV_64F).var() < 10:
                return False
        return True

    def _passes_occlusion_symmetry(self, kps):
        """
        Checks for facial symmetry and mouth-eye distance to detect occlusion.
        """
        left_eye = np.array(kps[0])
        right_eye = np.array(kps[1])
        nose = np.array(kps[2])
        mouth_left = np.array(kps[3])
        mouth_right = np.array(kps[4])

        face_width = np.linalg.norm(right_eye - left_eye)
        eye_mouth_dist = np.linalg.norm((mouth_left + mouth_right) / 2 - (left_eye + right_eye) / 2)

        # Symmetry: difference in nose distance from each eye
        symmetry_diff = abs(np.linalg.norm(left_eye - nose) - np.linalg.norm(right_eye - nose))
        if symmetry_diff > 0.4 * face_width:
            return False

        # Mouth should not be too close to eyes (occlusion)
        if eye_mouth_dist < 0.3 * face_width:
            return False

        return True
