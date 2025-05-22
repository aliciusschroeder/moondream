"""
Event handlers for the Gradio UI.
"""
import os
import gradio as gr
from PIL import Image

from ..core.model_loader import load_or_get_cached_model
from ..tasks.query import query_moondream_model
from ..tasks import placeholder_task_handler


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
    
    # UI-level validation: Use gr.Warning for recoverable issues, actual call raises gr.Error
    if pil_image is None:
        gr.Warning("Please upload an image for the query.")
        # Return values to clear/indicate error in result fields
        return (
            None,
            task_name,
            question_text or "N/A",
            "Error: Image required for query. Please upload an image.",
        )
    if not question_text or not question_text.strip():
        gr.Warning("Please enter a question for the query.")
        return (
            pil_image,
            task_name,
            "No question provided.",
            "Error: Question required for query. Please enter a question.",
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
        return pil_image, task_name, question_text, answer

    except gr.Error as ge:
        # Gradio will display this error automatically.
        # We still return values to update the result fields appropriately, showing the error message.
        print(f"Gradio Error caught in process_query_submission: {ge}")
        return pil_image, task_name, question_text, f"Operation Failed: {str(ge)}"
    except Exception as e:
        # Catch any other unexpected errors in this wrapper that weren't converted to gr.Error
        error_msg = f"An unexpected error occurred during query processing: {str(e)}"
        print(f"ERROR: {error_msg}")
        # Optionally, re-raise as gr.Error or return error message
        return pil_image, task_name, question_text, error_msg

