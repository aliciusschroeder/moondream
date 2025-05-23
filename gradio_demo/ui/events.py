"""
Event handlers for the Gradio UI.
"""

import os
import gradio as gr
from PIL import Image


from ..core.model_loader import load_or_get_cached_model


def handle_model_selection_change(selected_model_path: str):
    """
    Handle model selection change event.

    Args:
        selected_model_path (str): Path to the selected model file

    Yields:
        str: Status message updates as generator
        gr.update: Visibility update for load model button
    """
    if not selected_model_path:
        yield "⚠️ No model selected. Please choose a model.", gr.update()
        return

    yield f"⏳ Loading model: {os.path.basename(selected_model_path)}...", gr.update(visible=True)
    try:
        load_or_get_cached_model(selected_model_path)
        yield f"✅ Model '{os.path.basename(selected_model_path)}' loaded successfully.", gr.update(visible=False)
    except gr.Error as ge:
        yield f"❌ Error loading model: {str(ge)}", gr.update(visible=True)
    except Exception as e:
        yield f"❌ Failed to load model '{os.path.basename(selected_model_path)}': An unexpected error occurred: {str(e)}", gr.update(visible=True)


def handle_submission_error(
    error_message: str,
    image_col_update: gr.update,
    text_col_update: gr.update,
):
    """
    Handle submission error.

    Args:
        error_message (str): Error message to display
        image_col_update (gr.update): Update for the image column
        text_col_update (gr.update): Update for the text column

    Returns:
        tuple: (image=None, task_name, prompt, error message, image_col_update, text_col_update)
    """

    return (
        None,
        "Error",
        error_message,
        error_message,
        image_col_update,
        text_col_update,
    )


def check_submission(
    model_path_selected: str,
    pil_image: Image.Image,
    question_text: str,
):
    if pil_image is None:
        gr.Warning("Please upload an image for the query.")
        # Return values to clear/indicate error in result fields
        return "Error: Image required for query. Please upload an image."
    if not question_text or not question_text.strip():
        gr.Warning("Please enter a question for the query.")
        return "Error: Question required for query. Please enter a question."

    if not model_path_selected:
        gr.Warning("No model selected for query. Please choose a model from Settings.")
        return (
            "Error: No model selected for query. Please choose a model from Settings."
        )

    return None
