# Configuration file for Wedding Image Selector.
# This file centralizes all tunable parameters and thresholds used throughout the


# Sections:
#     - Face & Person Detection Thresholds
#     - Background Clutter Filtering
#     - Deduplication Settings
#     - Pose & Landmark Constraints
#     - Orientation & Override Rules
#     - Dancing/Party Filtering
#     - Composition Rejection Logic
#     - Emotion Override Logic
#     - Exposure Thresholds
#     - Face Size Variance
#     - Scoring Weights
#     - Output & Logging
#     - Final Selection Threshold
# """

# === Face & Person Detection Thresholds ===

SHARPNESS_THRESHOLD = 10  # Minimum Laplacian variance for sharpness
BRIGHTNESS_THRESHOLD = (30, 240)  # Acceptable mean brightness range for faces
MIN_EYE_OPEN_RATIO = 0.08  # Minimum eye openness (accounts for blinking/sunglasses)
CROPPED_FACE_RATIO_THRESHOLD = 0.4  # Max fraction of faces allowed to be cropped
VALID_FACE_RATIO_THRESHOLD = 0.2  # Lowered from 0.4 for debugging
CROPPED_FACE_MARGIN = 0.05  # Margin (as fraction of image size) to flag cropped faces

# === Background Clutter Filtering ===

BACKGROUND_CLUTTER_THRESHOLD = 0.6  # Max allowed clutter score (0=clean, 1=chaotic)
BACKGROUND_FACE_AREA_RATIO = 0.25  # Ignore faces < this ratio of main face area in solo/couple

# === Deduplication Settings ===

ENABLE_DEDUPLICATION = True  # Enable/disable deduplication step
DEDUP_THRESHOLD = 3  # Max Hamming distance for perceptual hash (phash)
DEDUP_USE_FACE_CLUSTERING = True  # Use face embedding clustering for deduplication
DEDUP_FACE_DISTANCE_THRESHOLD = 0.35  # Max cosine distance for face embedding similarity

# === Pose & Landmark Constraints ===

MAX_POSE_YAW = 40  # Max allowed yaw (left/right turn) in degrees
MAX_POSE_PITCH = 30  # Max allowed pitch (up/down tilt) in degrees
MIN_LANDMARKS_REQUIRED = 1         # Min number of detected landmarks to consider a valid face

# === Orientation & Override Rules ===

REJECT_GROUP_PORTRAIT = True  # Reject group photos in portrait orientation
REJECT_SOLO_COUPLE_ULTRAWIDE = True  # Reject solo/couple in extreme landscape
ORIENTATION_OVERRIDE_CRITERIA = {
    "aesthetic_score": 0.6,      # Allow override if aesthetic score is high
    "composition_score": 0.5     # ...or if composition is good
}

# === Dancing/Party Filtering ===

DANCING_FILTER_CRITERIA = {
    "emotion_score": 0.5,        # Min emotion score to consider as dancing/party
    "max_center_score": 0.5,     # Max allowed center score (should not be well-centered)
    "max_thirds_score": 0.5      # Max allowed thirds score (should not be well-composed)
}

# === Composition Rejection Logic ===

COMPOSITION_MIN_THRESHOLD = 0.3  # Min score for any composition metric
COMPOSITION_FAIL_COUNT_LIMIT = 2  # Max allowed metrics below threshold before rejection

# === Emotion Override Logic ===

EMOTION_OVERRIDE_CRITERIA = {
    "emotion_score": 0.6,        # Min emotion score for override
    "aesthetic_score": 0.6,      # Min aesthetic score for override
    "sharpness_score": 0.5,      # Min sharpness score for override
    "exposure_score": 0.5,       # Min exposure score for override
}

# === Exposure Thresholds ===

UNDEREXPOSED_THRESHOLD = 0.12  # Max fraction of pixels considered underexposed
OVEREXPOSED_THRESHOLD = 0.12   # Max fraction of pixels considered overexposed

# === Face Size Variance ===

FACE_SIZE_VARIANCE_THRESHOLD = 0.25  # Max normalized variance among valid face sizes

# === Scoring Weights (Weighted Average) ===

SCORE_WEIGHTS = {
    "aesthetic_score": 3.0,         # Visual appeal
    "center_score": 1.5,            # Center alignment
    "thirds_score": 1.2,            # Rule of thirds
    "spiral_score": 1.0,            # Golden spiral
    "symmetry_score": 1.0,          # Horizontal symmetry
    "color_harmony_score": 1.3,     # Color harmony
    "emotion_score": 1.2,           # Detected emotion
    "eye_open_score": 1.0,          # Eye openness
    "sharpness_score": 1.0,         # Overall sharpness
    "exposure_score": 1.0           # Exposure quality
}

# === Output & Logging ===

TOP_N_IMAGES = 300  
CSV_LOGGING_ENABLED = True  
CSV_OUTPUT_FILENAME = "image_scores.csv" 

# === Final Selection Threshold ===

FINAL_SCORE_THRESHOLD = 0.3  # Images below this score are always rejected