import hashlib


def get_image_hash(image):
    """Generate a simple hash for an image to detect changes"""
    if image is None:
        return None
    try:
        # Get a simplified representation of the image
        img_copy = image.copy()
        img_copy.thumbnail((100, 100))  # Resize for faster hashing
        img_bytes = img_copy.tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    except Exception:
        # Fallback to None if we can't hash the image
        return None
