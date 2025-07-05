

import cv2
import numpy as np
from PIL import Image
import imagehash
from insightface.app import FaceAnalysis


class ImageDeduplicator:
    """
    Deduplicates images using perceptual hashing and (optionally) face embedding similarity.
    """

    def __init__(
        self,
        hash_func: str = "phash",
        threshold: int = 5,
        use_face_clustering: bool = True,
        face_distance_threshold: float = 0.4
    ):
        """
        Initializes the deduplicator.

        Args:
            hash_func (str): Name of the imagehash function to use (e.g., "phash").
            threshold (int): Max hash distance to consider images as duplicates.
            use_face_clustering (bool): Whether to use face embeddings for deduplication.
            face_distance_threshold (float): Max embedding distance for face similarity.
        """
        self.hash_func = getattr(imagehash, hash_func)
        self.threshold = threshold
        self.use_face_clustering = use_face_clustering
        self.face_distance_threshold = face_distance_threshold

        # Initialize face analyzer if needed
        self.face_analyzer = None
        if self.use_face_clustering:
            self._init_face_analyzer()

        # Internal caches for efficiency
        self._hash_cache = {}
        self._embedding_cache = {}

    def _init_face_analyzer(self):
        """Initializes the InsightFace analyzer."""
        self.face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    def get_hash(self, image_path: str):
        """
        Computes or retrieves the perceptual hash for an image.

        Args:
            image_path (str): Path to the image.

        Returns:
            imagehash.ImageHash or None: The computed hash, or None on failure.
        """
        if image_path in self._hash_cache:
            return self._hash_cache[image_path]
        try:
            img = Image.open(image_path).convert("RGB")
            img_hash = self.hash_func(img)
            self._hash_cache[image_path] = img_hash
            return img_hash
        except Exception:
            return None

    def get_embedding(self, image_path: str):
        """
        Computes or retrieves the face embedding for the largest face in an image.

        Args:
            image_path (str): Path to the image.

        Returns:
            np.ndarray or None: The face embedding, or None if not found.
        """
        if image_path in self._embedding_cache:
            return self._embedding_cache[image_path]
        if not self.face_analyzer:
            return None
        try:
            bgr = cv2.imread(image_path)
            if bgr is None:
                return None
            faces = self.face_analyzer.get(bgr)
            if not faces:
                return None
            main_face = self._get_largest_face(faces)
            emb = main_face.embedding
            self._embedding_cache[image_path] = emb
            return emb
        except Exception:
            return None

    @staticmethod
    def _get_largest_face(faces):
        """
        Returns the face with the largest bounding box area.

        Args:
            faces (list): List of face objects.

        Returns:
            Face object: The largest face.
        """
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def _hash_similar(self, img1_path: str, img2_path: str) -> bool:
        """
        Checks if two images are similar based on perceptual hash.

        Returns:
            bool: True if similar, False otherwise.
        """
        h1 = self.get_hash(img1_path)
        h2 = self.get_hash(img2_path)
        if h1 is not None and h2 is not None:
            hash_diff = abs(h1 - h2)
            return hash_diff <= self.threshold
        return False

    def _face_embedding_similar(self, img1_path: str, img2_path: str) -> bool:
        """
        Checks if two images are similar based on face embedding distance.

        Returns:
            bool: True if similar, False otherwise.
        """
        if not self.use_face_clustering:
            return False
        e1 = self.get_embedding(img1_path)
        e2 = self.get_embedding(img2_path)
        if e1 is not None and e2 is not None:
            dist = np.linalg.norm(e1 - e2)
            return bool(dist < self.face_distance_threshold)
        return False

    def are_similar(self, img1_path: str, img2_path: str) -> bool:
        """
        Determines if two images are visually or semantically similar.

        Args:
            img1_path (str): Path to the first image.
            img2_path (str): Path to the second image.

        Returns:
            bool: True if images are considered similar, False otherwise.
        """
        # First, try perceptual hash
        if self._hash_similar(img1_path, img2_path):
            return True

        # Then, try face embedding if enabled
        if self._face_embedding_similar(img1_path, img2_path):
            return True

        return False
