# resizer.py
import cv2 as cv

def resize_image(image, scale_factor):
    if image is None:
        raise ReferenceError("No image detected")

    scale_factor = 0.5

    # Resize the image using scaling factors fx and fy
    # INTER_AREA is generally recommended for shrinking images
    scaled_img = cv.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)

    return scaled_img