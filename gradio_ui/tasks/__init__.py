"""
Task initialization and management module.
"""

from .query import query_moondream_model
from .suggestions import get_question_suggestions, get_object_suggestions
from .caption import caption_image
from .point import point_objects
from .detect import detect_objects
from .detect_all import detect_all_objects

__all__ = [
    "query_moondream_model",
    "get_question_suggestions",
    "get_object_suggestions",
    "caption_image",
    "point_objects",
    "detect_objects",
    "detect_all_objects",
]
