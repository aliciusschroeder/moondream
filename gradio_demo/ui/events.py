"""
Event handlers for the Gradio UI.
"""

import os
import gradio as gr
from PIL import Image

from ..core.model_loader import load_or_get_cached_model
from ..tasks.suggestions import get_question_suggestions
from ..tasks.query import query_moondream_model
from ..utils.img_utils import get_image_hash


def handle_model_selection_change(selected_model_path: str):
    """
    Handle model selection change event.

    Args:
        selected_model_path (str): Path to the selected model file

    Yields:
        str: Status message updates as generator
    """
    if not selected_model_path:
        yield "⚠️ No model selected. Please choose a model."
        return

    yield f"⏳ Loading model: {os.path.basename(selected_model_path)}..."
    try:
        load_or_get_cached_model(selected_model_path)
        yield f"✅ Model '{os.path.basename(selected_model_path)}' loaded successfully."
    except gr.Error as ge:
        yield f"❌ Error loading model: {str(ge)}"
    except Exception as e:
        yield f"❌ Failed to load model '{os.path.basename(selected_model_path)}': An unexpected error occurred: {str(e)}"


def generate_question_suggestions(
    model_path_selected, image, last_processed_image, current_tab
):
    """
    Generate question suggestions for the current image.

    Args:
        model_path_selected (str): Path to the selected model
        image (PIL.Image): The current image
        last_processed_image (str): Hash of the last processed image
        current_tab (str): The current tab ID

    Returns:
        tuple: Updated UI elements and state
    """
    print(f"Current tab: {current_tab}")
    # Check if we need to generate new suggestions
    if current_tab != "query_tab" or image is None:
        return (
            gr.update(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            last_processed_image,
        )

    # Get hash of current image
    current_hash = get_image_hash(image)

    # Check if image changed or first time
    if current_hash == last_processed_image:
        # Image hasn't changed, keep existing suggestions
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            last_processed_image,
        )

    # Image changed or first time, generate new suggestions
    try:
        # Show loading status
        status_msg = "*Generating question suggestions...*"

        # Get suggestions from model
        suggestions = get_question_suggestions(model_path_selected, image)

        # Update buttons with suggestions
        return (
            gr.update(value="", visible=False),  # Hide status
            gr.update(value=suggestions[0], visible=True),
            gr.update(value=suggestions[1], visible=True),
            gr.update(value=suggestions[2], visible=True),
            None,  # No need to update question textbox yet
            current_hash,  # Update processed image hash
        )
    except gr.Error as e:
        # Error occurred, show message and hide buttons
        return (
            gr.update(value=f"*{str(e)}*", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            last_processed_image,  # Keep old hash
        )
    except Exception as e:
        # Unexpected error, show message and hide buttons
        error_msg = f"*Error generating suggestions: {str(e)}*"
        return (
            gr.update(value=error_msg, visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            last_processed_image,  # Keep old hash
        )


def handle_suggestion_click(suggestion_text):
    """
    Handle when a suggestion button is clicked.

    Args:
        suggestion_text (str): The text from the clicked suggestion button

    Returns:
        str: The suggestion text to fill the query input
    """
    return suggestion_text


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
        "N/A",
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


def process_query_submission(
    model_path_selected: str,
    pil_image: Image.Image,
    question_text: str,
    max_tokens_val: int,
    temperature_val: float,
    top_p_val: float,
):
    """
    Process query task submission.

    Args:
        model_path_selected (str): Path to the model file
        pil_image (PIL.Image): Image to query
        question_text (str): Question to ask about the image
        max_tokens_val (int): Maximum tokens to generate
        temperature_val (float): Temperature for generation
        top_p_val (float): Top-p value for generation

    Returns:
        tuple: (image, task_name, prompt, response)
    """
    task_name = "Query"
    image_col_update = gr.update(scale=1)
    text_col_update = gr.update(scale=7)

    # Check for submission errors
    error_message = check_submission(
        model_path_selected,
        pil_image,
        question_text,
    )
    if error_message:
        return handle_submission_error(
            error_message,
            image_col_update,
            text_col_update,
        )

    try:
        # query_moondream_model will raise gr.Error on failure (model load, core query error)
        answer = query_moondream_model(
            model_path_selected,
            pil_image,
            question_text,
            max_tokens_val,
            temperature_val,
            top_p_val,
        )
        return (
            pil_image,
            task_name,
            question_text,
            answer,
            image_col_update,
            text_col_update,
        )

    except gr.Error as ge:
        # Gradio will display this error automatically.
        # We still return values to update the result fields appropriately, showing the error message.
        print(f"Gradio Error caught in process_query_submission: {ge}")
        return handle_submission_error(
            f"Operation Failed: {str(ge)}",
            image_col_update,
            text_col_update,
        )
    except Exception as e:
        # Catch any other unexpected errors in this wrapper that weren't converted to gr.Error
        error_msg = f"An unexpected error occurred during query processing: {str(e)}"
        print(f"ERROR: {error_msg}")
        # Optionally, re-raise as gr.Error or return error message
        return handle_submission_error(
            error_msg,
            image_col_update,
            text_col_update,
        )
