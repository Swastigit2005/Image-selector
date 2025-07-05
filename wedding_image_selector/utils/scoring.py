

import cv2
import numpy as np
from models.emotion_recognizer import EmotionRecognizer
from models.person_detector import PersonDetector
from models.face_detector import FaceDetector
from models.face_quality import FaceQualityScorer
from models.aesthetic_score import AestheticScorer
from models.composition_score import CompositionScorer
from models.background_quality import BackgroundClutterScorer
from utils.image_utils import compress_image
import config
from utils.face_detector import InsightFaceDetector

class ImageScorer:
    def __init__(self):
        self.person_detector = PersonDetector()
        self.face_detector = FaceDetector()
        self.face_quality = FaceQualityScorer()
        self.aesthetic_scorer = AestheticScorer()
        self.background_scorer = BackgroundClutterScorer()
        self.emotion_recognizer = EmotionRecognizer()
        self.insight_face_detector = InsightFaceDetector()

    def score_image(self, image_path):
        """
        Main entry point: scores an image and returns a result dict.
        """
        result = self._init_result(image_path)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] Could not load image: {image_path}")
                return self._reject(result, "Unreadable image")

            image = compress_image(image)
            h, w = image.shape[:2]

            # Person detection
            if not self.person_detector.has_real_person(image_path):
                return self._reject(result, "No person detected")

            # Face detection and filtering
            faces = self.face_detector.detect_faces(image_path)
            result["num_faces"] = len(faces)
            if len(faces) == 0:
                return self._reject(result, "No faces detected")

            result["image_type"] = self._get_image_type(faces)
            faces = self._filter_background_faces(faces, result["image_type"])
            valid_faces = self._get_valid_faces(faces, image, result["image_type"])
            result["face_passed"] = len(valid_faces)
            if not self._has_enough_valid_faces(valid_faces, faces):
                return self._reject(result, "Too few valid faces")

            if self._too_many_cropped_faces(faces):
                return self._reject(result, "Too many cropped faces")

            # Background clutter
            if not self._check_background_clutter(image, result):
                return result

            # Composition
            scorer = CompositionScorer(image.shape)
            face_centers = scorer.get_face_centers_for_scoring(image)
            print(f"[DEBUG] Face centers used for scoring: {face_centers}")

            if not face_centers:
                print(f"[ERROR] No valid faces detected for image: {image_path}")

            # --- VALIDATION LOGIC ---
            # If you require at least 1 face:
            if not face_centers or len(face_centers) < 1:
                return {"valid": False, "rejection_reason": "Too few valid faces"}

            
            if not face_centers:
                print(f"[ERROR] No valid faces detected for image: {image_path}")

            
            center_score = scorer.face_center_score(face_centers)
            thirds_score = scorer.rule_of_thirds_score(face_centers)
            spiral_score = scorer.golden_spiral_score(face_centers)
            symmetry_score = scorer.symmetry_score(face_centers)
            color_harmony_score = scorer.color_harmony_score(image)
            
            result = {
                "center_score": center_score,
                "thirds_score": thirds_score,
                "spiral_score": spiral_score,
                "symmetry_score": symmetry_score,
                "color_harmony_score": color_harmony_score,
                "valid": True,
                "rejection_reason": None,
                
            }
        

            # Orientation rules
            if not self._check_orientation_rules(result, h, w):
                return result

            # Sharpness and exposure
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self._compute_sharpness(result, gray)
            if not self._compute_exposure(result, gray):
                return result

            # Emotion and aesthetic
            self._compute_emotion(result, image)
            self._compute_aesthetic(result, image_path)

            # Emotion override
            emotion_override = self._check_emotion_override(result)

            # Composition rejection
            if self._should_reject_composition(result, emotion_override):
                return result

            # Dancing filter
            if self._should_reject_dancing(result, valid_faces, emotion_override):
                return result

            # Face size variance
            if self._should_reject_face_size_variance(result, valid_faces, emotion_override):
                return result

            # Eye open score
            self._compute_eye_open_score(result, valid_faces)

            # Final score
            self._compute_final_score(result)
            result["valid"] = True

        except Exception as e:
            result["rejection_reason"] = f"Exception: {e}"

        return result

    # === Helper methods below ===

    def _init_result(self, image_path):
        """Initializes the result dictionary."""
        return {
            "path": image_path,
            "valid": False,
            "image_type": "",
            "num_faces": 0,
            "face_passed": 0,
            "aesthetic_score": 0.0,
            "center_score": 0.0,
            "thirds_score": 0.0,
            "spiral_score": 0.0,
            "symmetry_score": 0.0,
            "color_harmony_score": 0.0,
            "emotion_score": 0.0,
            "eye_open_score": 0.0,
            "sharpness_score": 0.0,
            "exposure_score": 0.0,
            "final_score": 0.0,
            "background_clutter_score": 0.0,
            "rejection_reason": ""
        }

    def _reject(self, result, reason):
        #Sets rejection reason and returns result.
        result["rejection_reason"] = reason
        return result

    def _get_image_type(self, faces):
        #Determines image type based on number of faces.
        if len(faces) == 1:
            return "solo"
        elif len(faces) == 2:
            return "couple"
        else:
            return "group"

    def _filter_background_faces(self, faces, image_type):
        #Removes background faces for solo/couple images.
        if image_type in ("solo", "couple") and len(faces) > 1:
            areas = [(i, (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                     for i, f in enumerate(faces)]
            areas.sort(key=lambda x: x[1], reverse=True)
            main_area = areas[0][1]
            return [faces[idx] for idx, area in areas if area >= config.BACKGROUND_FACE_AREA_RATIO * main_area]
        return faces

    def _get_valid_faces(self, faces, image, image_type):
        """Returns faces that pass quality checks."""
        return [f for f in faces if self.face_quality.is_face_good(f, image, image_type)]

    def _has_enough_valid_faces(self, valid_faces, faces):
        """Checks if there are enough valid faces."""
        return len(valid_faces) >= max(1, int(len(faces) * config.VALID_FACE_RATIO_THRESHOLD))

    def _too_many_cropped_faces(self, faces):
        """Checks if too many faces are cropped."""
        cropped_count = sum(1 for f in faces if f.get("is_cropped"))
        return cropped_count > config.CROPPED_FACE_RATIO_THRESHOLD * len(faces)

    def _check_background_clutter(self, image, result):
        """Computes and checks background clutter score."""
        bg_score = self.background_scorer.compute_clutter_score(image)
        result["background_clutter_score"] = round(bg_score, 2)
        if bg_score > config.BACKGROUND_CLUTTER_THRESHOLD:
            result["rejection_reason"] = f"Cluttered background (score={bg_score:.2f})"
            return False
        return True

    def _compute_composition_scores(self, result, comp, face_centers, image):
        """Computes all composition-related scores."""
        result["center_score"] = comp.face_center_score(face_centers)
        result["thirds_score"] = comp.rule_of_thirds_score(face_centers)
        result["spiral_score"] = comp.golden_spiral_score(face_centers)
        result["symmetry_score"] = comp.symmetry_score(face_centers)
        result["color_harmony_score"] = comp.color_harmony_score(image)

    def _check_orientation_rules(self, result, h, w):
        """Checks orientation rules and rejects if needed."""
        orientation_issue = None
        image_type = result["image_type"]
        if config.REJECT_GROUP_PORTRAIT and image_type == "group" and h > w:
            orientation_issue = "Group photo in portrait orientation"
        elif config.REJECT_SOLO_COUPLE_ULTRAWIDE and image_type in ("solo", "couple") and w > 2 * h:
            orientation_issue = "Solo/Couple in extreme landscape orientation"

        if orientation_issue:
            comp_score = np.mean([
                result["center_score"],
                result["thirds_score"],
                result["spiral_score"],
                result["symmetry_score"]
            ])
            if result["aesthetic_score"] < config.ORIENTATION_OVERRIDE_CRITERIA["aesthetic_score"] or \
               comp_score < config.ORIENTATION_OVERRIDE_CRITERIA["composition_score"]:
                result["rejection_reason"] = f"{orientation_issue} with poor composition/aesthetic"
                return False
        return True

    def _compute_sharpness(self, result, gray):
        """Computes sharpness score."""
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        result["sharpness_score"] = round(min(sharpness / 1000.0, 1.0), 2)

    def _compute_exposure(self, result, gray):
        """Computes exposure score and checks for exposure issues."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total = gray.size
        under_exp = np.sum(hist[:int(config.UNDEREXPOSED_THRESHOLD * 256)]) / total
        over_exp = np.sum(hist[int(256 - config.OVEREXPOSED_THRESHOLD * 256):]) / total
        exposure_score = 1.0 - (under_exp + over_exp)
        result["exposure_score"] = round(max(min(float(exposure_score), 1.0), 0.0), 2)
        if under_exp > config.UNDEREXPOSED_THRESHOLD or over_exp > config.OVEREXPOSED_THRESHOLD:
            result["rejection_reason"] = f"Exposure issues (under={under_exp:.2f}, over={over_exp:.2f})"
            return False
        return True

    def _compute_emotion(self, result, image):
        """Computes emotion score."""
        try:
            result["emotion_score"] = self.emotion_recognizer.get_emotion_score(image)
        except Exception:
            result["emotion_score"] = 0.0

    def _compute_aesthetic(self, result, image_path):
        """Computes aesthetic score."""
        result["aesthetic_score"] = self.aesthetic_scorer.score(image_path)

    def _check_emotion_override(self, result):
        """Checks if emotion override criteria are met."""
        c = config.EMOTION_OVERRIDE_CRITERIA
        return (
            c and
            result["emotion_score"] >= c["emotion_score"] and
            result["aesthetic_score"] >= c["aesthetic_score"] and
            result["sharpness_score"] >= c["sharpness_score"] and
            result["exposure_score"] >= c["exposure_score"] and
            result["face_passed"] >= 1
        )

    def _should_reject_composition(self, result, emotion_override):
        """Rejects if too many composition scores are low."""
        failed = [
            k for k in ["center_score", "thirds_score", "spiral_score", "symmetry_score", "color_harmony_score"]
            if result[k] < config.COMPOSITION_MIN_THRESHOLD
        ]
        if len(failed) > config.COMPOSITION_FAIL_COUNT_LIMIT and not emotion_override:
            result["rejection_reason"] = f"Low composition: {', '.join(failed)}"
            return True
        return False

    def _should_reject_dancing(self, result, valid_faces, emotion_override):
        """Rejects group dancing images lacking clarity/composition."""
        if result["image_type"] == "group" and len(valid_faces) >= 3:
            c = config.DANCING_FILTER_CRITERIA
            is_dancing = (
                result["emotion_score"] >= c["emotion_score"] and
                result["center_score"] < c["max_center_score"] and
                result["thirds_score"] < c["max_thirds_score"]
            )
            if is_dancing and not emotion_override:
                result["rejection_reason"] = "Group dancing image lacks clarity/composition"
                return True
        return False

    def _should_reject_face_size_variance(self, result, valid_faces, emotion_override):
        """Rejects images with high face size variance and low emotion/composition."""
        if len(valid_faces) >= 3:
            sizes = [(f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]) for f in valid_faces]
            mean_size = np.mean(sizes)
            variance = np.var(sizes) / (mean_size**2 + 1e-6)
            min_comp = min(
                result["center_score"],
                result["thirds_score"],
                result["spiral_score"],
                result["symmetry_score"]
            )
            if variance > config.FACE_SIZE_VARIANCE_THRESHOLD and not emotion_override:
                if result["emotion_score"] < 0.4 or min_comp < config.COMPOSITION_MIN_THRESHOLD:
                    result["rejection_reason"] = (
                        f"High face size variance with low emotion/composition "
                        f"(var={round(variance, 2)}, emo={result['emotion_score']}, comp={round(min_comp, 2)})"
                    )
                    return True
        return False

    def _compute_eye_open_score(self, result, valid_faces):
        """Computes average eye open score for valid faces."""
        try:
            scores = []
            for f in valid_faces:
                lm = f.get("landmarks", [])
                if len(lm) >= 2:
                    eye_height = abs(lm[1][1] - lm[0][1])
                    eye_width = np.linalg.norm(np.array(lm[1]) - np.array(lm[0])) + 1e-6
                    eye_ratio = eye_height / eye_width
                    scores.append(eye_ratio)
            if scores:
                avg = float(np.mean(scores))
                result["eye_open_score"] = round(min(max(avg, 0.0), 1.0), 2)
        except Exception:
            result["eye_open_score"] = 0.0

    def _compute_final_score(self, result):
        """Computes the weighted final score."""
        weights = config.SCORE_WEIGHTS
        result["final_score"] = round(
            sum(result[k] * w for k, w in weights.items()) / sum(weights.values()), 3
        )
        """Computes the weighted final score."""
        weights = config.SCORE_WEIGHTS
        result["final_score"] = round(
            sum(result[k] * w for k, w in weights.items()) / sum(weights.values()), 3
        )
