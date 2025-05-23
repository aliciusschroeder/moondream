"""
Task initialization and management module.
"""

from .query import query_moondream_model
from .suggestions import get_question_suggestions, get_object_suggestions
from .caption import caption_image

__all__ = [
    "query_moondream_model",
    "placeholder_task_handler",
    "get_question_suggestions",
    "caption_image",
]
