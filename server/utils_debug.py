import cv2
import base64
import numpy as np

def save_img(image, name: str):
    """Save either a NumPy (BGR) image or a Base64-encoded image to disk."""
    if isinstance(image, np.ndarray):
        cv2.imwrite(name, image)
        print(f"Saved NumPy image to {name}")
    elif isinstance(image, str):
        # Assume Base64-encoded bytes
        image_data = base64.b64decode(image)
        with open(name, "wb") as f:
            f.write(image_data)
        print(f"Saved Base64 image to {name}")
    else:
        raise TypeError("Unsupported image type — expected NumPy array or Base64 string")

