"""
Task-specific logic for querying the Moondream model.
"""

import gradio as gr
from PIL import Image

from ..core.model_loader import load_or_get_cached_model


def query_moondream_model(
    model_path_selected: str,
    pil_image: Image.Image,
    question_text: str,
    max_tokens_val: int,
    temperature_val: float,
    top_p_val: float,
):
    """
    Query the Moondream model with an image and question.

    Args:
        model_path_selected (str): Path to the model file
        pil_image (PIL.Image): Image to query
        question_text (str): Question to ask about the image
        max_tokens_val (int): Maximum tokens to generate
        temperature_val (float): Temperature for generation
        top_p_val (float): Top-p value for generation

    Returns:
        str: The model's response

    Raises:
        gr.Error: If inputs are invalid or query fails
    """
    # Input validation
    if not model_path_selected:
        raise gr.Error(
            "No model selected for query. Please choose a model from Settings."
        )
    if pil_image is None:
        raise gr.Error("No image provided for query. Please upload an image.")
    if not question_text or not question_text.strip():
        raise gr.Error("No question provided for query. Please enter a question.")

    try:
        model = load_or_get_cached_model(model_path_selected)

        # Model should not be None if load_or_get_cached_model didn't raise gr.Error
        # but defensive check is okay if there's any doubt.
        if model is None:
            raise gr.Error(
                "Model could not be loaded. Check application logs and settings."
            )

        print(f"Querying model with question: '{question_text}'...")
        text_sampling_settings = {
            "max_tokens": int(max_tokens_val),
            "temperature": float(temperature_val),
            "top_p": float(top_p_val),
        }
        print(f"Using text sampling settings: {text_sampling_settings}")

        result_dict = model.query(
            image=pil_image,
            question=question_text,
            stream=False,
            settings=text_sampling_settings,
        )

        answer = result_dict.get("answer", "No answer returned by the model.")
        print(f"Model responded: '{answer}'")
        return answer  # Return only the answer string for success

    except gr.Error:  # Re-raise gr.Error exceptions to let Gradio handle them in the UI
        raise
    except Exception as e:  # Catch other unexpected errors
        error_message = f"An unexpected error occurred during the model query: {str(e)}"
        print(f"ERROR: {error_message}")
        # Convert other exceptions to gr.Error for consistent UI error reporting
        raise gr.Error(error_message)
