"""
Wrapper for importing Moondream classes with fallback logic (cleanly encapsulated).
"""

# --- Moondream Imports ---
# This section attempts to import Moondream components with fallbacks if imports fail
try:
    from moondream.torch.moondream import (
        MoondreamModel,
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
        DEFAULT_MAX_OBJECTS,
    )
    from moondream.torch.config import MoondreamConfig
    from moondream.torch.weights import load_weights_into_model

    # Set flag indicating successful import
    MOONDREAM_IMPORTS_SUCCESS = True

except ImportError as e:
    print("ERROR: Failed to import Moondream components.")
    print(
        "Please ensure the Moondream library is correctly installed and accessible in your PYTHONPATH."
    )
    print(f"Details: {e}")

    # Set flag indicating import failure
    MOONDREAM_IMPORTS_SUCCESS = False

    # Define fallback dummy constants
    DEFAULT_MAX_TOKENS = 512  # Fallback default
    DEFAULT_TEMPERATURE = 0.2  # Fallback default
    DEFAULT_TOP_P = 1.0  # Fallback default
    DEFAULT_MAX_OBJECTS = 10  # Fallback default

    # Define dummy classes if import fails
    class MoondreamConfig:
        """Dummy MoondreamConfig class when imports fail."""

    class MoondreamModel:
        """Dummy MoondreamModel class when imports fail."""

        def __init__(self, config):
            self.config = config

        def query(self, image, question, stream=False, settings=None):
            return {"answer": "Error: MoondreamModel not loaded due to import failure."}

        def to(self, device):
            pass

        def eval(self):
            pass

    def load_weights_into_model(weights_file, model):
        """Dummy load_weights function when imports fail."""
        raise ImportError("load_weights_into_model is not available.")


# Make components available for import from this module
__all__ = [
    "MoondreamModel",
    "MoondreamConfig",
    "load_weights_into_model",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_MAX_OBJECTS",
    "MOONDREAM_IMPORTS_SUCCESS",
]
