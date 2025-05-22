"""
Task initialization and management module.
"""
from .query import query_moondream_model
from ..core.placeholder_tasks import placeholder_task_handler

__all__ = [
    'query_moondream_model',
    'placeholder_task_handler',
]

