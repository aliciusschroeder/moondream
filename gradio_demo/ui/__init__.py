"""
UI components and event handlers for the Moondream Gradio application.
"""

from .events_tasks import process_caption_submission, process_query_submission
from .layout import create_gradio_ui
from .events import handle_model_selection_change

__all__ = [
    "create_gradio_ui",
    "handle_model_selection_change",
    "process_query_submission",
    "process_caption_submission",
]
