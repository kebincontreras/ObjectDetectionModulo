
import cv2
import numpy as np

def modulo(x, L):
    positive = x > 0
    x = x % L
    x = np.where( ( x == 0) &  positive, L, x)
    return x

def apply_blur(image, kernel_size):
    """Aplica desenfoque gaussiano a la imagen."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def clip_image(image, correction, sat_factor):
    """Clips image with saturation factor and correction."""
    processed_image = np.power(image, correction) * sat_factor
    clipped_image = np.clip(processed_image, 0, 1)
    return clipped_image

def wrap_image(image, correction, sat_factor):
    """Wraps image with saturation factor and correction."""
    processed_image = np.power(image, 1.0) * sat_factor
    wrapped_image =  modulo(processed_image, 1.0)
    return wrapped_image


