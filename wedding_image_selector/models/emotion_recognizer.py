

import numpy as np
import cv2
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

class EmotionRecognizer:
    """
    Recognizes facial emotions and computes a happiness-focused emotion score.
    """

    def __init__(self, model_name='enet_b0_8_best_afew'):
        """
        Initializes the emotion recognizer with the specified model.
        """
        self.model = HSEmotionRecognizer(model_name=model_name)

    def get_emotion_score(self, face_crop):
        """
        Computes a weighted emotion score for a face crop.

        Args:
            face_crop (np.ndarray): Face-cropped image in BGR format.

        Returns:
            float: Weighted emotion score in [0.0, 1.0].
        """
        try:
            if not self._is_valid_face_crop(face_crop):
                return 0.0

            processed_crop = self._preprocess_face_crop(face_crop)
            emotions, scores = self._predict_emotions(processed_crop)
            score = self._compute_weighted_score(scores)
            return self._sanitize_score(score)
        except Exception as e:
            print(f"[EmotionRecognizer] Error: {e}")
            return 0.0

    def _is_valid_face_crop(self, face_crop):
        """
        Checks if the input is a valid face crop.

        Args:
            face_crop (np.ndarray): Input image.

        Returns:
            bool: True if valid, False otherwise.
        """
        return face_crop is not None and isinstance(face_crop, np.ndarray) and face_crop.size > 0

    def _preprocess_face_crop(self, face_crop):
        """
        Ensures the face crop is the correct size and color format.

        Args:
            face_crop (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Preprocessed RGB image.
        """
        h, w = face_crop.shape[:2]
        if h < 48 or w < 48:
            face_crop = cv2.resize(face_crop, (64, 64), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        return img_rgb

    def _predict_emotions(self, img_rgb):
        """
        Predicts emotions using the HSEmotion model.

        Args:
            img_rgb (np.ndarray): RGB face image.

        Returns:
            tuple: (emotions, scores)
        """
        return self.model.predict_emotions(img_rgb, logits=False)

    def _compute_weighted_score(self, scores):
        """
        Computes the weighted happiness-focused emotion score.

        Args:
            scores (list or np.ndarray): Emotion scores.

        Returns:
            float: Weighted score.
        """
        # Emotion order: [Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise]
        happy = scores[4]
        neutral = scores[5]
        surprise = scores[7]
        return (0.6 * happy) + (0.3 * surprise) + (0.1 * neutral)

    def _sanitize_score(self, score):
        """
        Clamps and rounds the score to [0.0, 1.0] with 2 decimals.

        Args:
            score (float): Raw score.

        Returns:
            float: Sanitized score.
        """
        return round(min(max(float(score), 0.0), 1.0), 2)
