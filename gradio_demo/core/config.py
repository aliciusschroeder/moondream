"""
Constants such as MODEL_DIR, default parameters, and app configuration.
"""
import os

# --- Configuration ---
# Path to directory containing model files
MODEL_DIR = os.environ.get("MOONDREAM_MODEL_DIR", "/home/alec/moondream/models/")

# Application display settings
APP_TITLE = "🌔 Moondream Interface"
APP_DESCRIPTION = """
Upload an image, configure settings, choose a task, provide input, and see the results.
Models are loaded from: `{}`.
""".format(MODEL_DIR)

# Default model selection message
DEFAULT_MODEL_STATUS_MESSAGE = "No models found or directory empty."

# Theme settings
PRIMARY_HUE = "blue"
SECONDARY_HUE = "sky"

# Debug mode
DEBUG_MODE = True

# Function to get all constants as a dict (useful for debugging/logging)
def get_config_dict():
    """Return all config constants as a dictionary."""
    return {
        "MODEL_DIR": MODEL_DIR,
        "APP_TITLE": APP_TITLE,
        "DEBUG_MODE": DEBUG_MODE,
    }

