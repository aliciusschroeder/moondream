"""
Core functionality for the Moondream Gradio application.
"""

from .config import MODEL_DIR, APP_TITLE, APP_DESCRIPTION, DEBUG_MODE, get_config_dict
from .model_loader import (
    get_model_files_from_directory,
    load_or_get_cached_model,
    initialize_model,
)
from .placeholder_tasks import placeholder_task_handler

__all__ = [
    "MODEL_DIR",
    "APP_TITLE",
    "APP_DESCRIPTION",
    "DEBUG_MODE",
    "get_config_dict",
    "get_model_files_from_directory",
    "load_or_get_cached_model",
    "initialize_model",
    "placeholder_task_handler",
]
