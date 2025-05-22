"""
Task initialization and management module.
"""

from .query import query_moondream_model
from ..core.placeholder_tasks import placeholder_task_handler
from .suggestions import get_question_suggestions

__all__ = [
    "query_moondream_model",
    "placeholder_task_handler",
    "get_question_suggestions",
]
