

import cv2
import numpy as np

def validate_image(image):
    """
    Validates that the input is a proper OpenCV image (numpy ndarray).
    Raises ValueError if invalid.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid input image for processing")

def calculate_new_dimensions(h, w, max_dim):
    """
    Calculates new dimensions for resizing while preserving aspect ratio.

    Args:
        h (int): Original image height.
        w (int): Original image width.
        max_dim (int): Maximum allowed size for the longer edge.

    Returns:
        (int, int): New width and height.
    """
    if max(h, w) <= max_dim:
        return w, h
    scale = max_dim / max(h, w)
    return int(w * scale), int(h * scale)

def resize_image(image, new_w, new_h):
    """
    Resizes the image to the specified dimensions using high-quality downsampling.

    Args:
        image (np.ndarray): Input image.
        new_w (int): New width.
        new_h (int): New height.

    Returns:
        np.ndarray: Resized image.
    """
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def compress_image(image, max_dim=768):
    """
    Compresses (resizes) the image so that the longer side is at most `max_dim` pixels.
    Preserves aspect ratio and uses high-quality downsampling.

    Args:
        image (np.ndarray): Input image (BGR format).
        max_dim (int): Maximum size for the longer edge. Default is 768.

    Returns:
        np.ndarray: Compressed (resized) image.
    """
    validate_image(image)
    h, w = image.shape[:2]
    new_w, new_h = calculate_new_dimensions(h, w, max_dim)
    if (new_w, new_h) == (w, h):
        return image
    return resize_image(image, new_w, new_h)
