from ..tasks.caption import caption_image
from ..tasks.query import query_moondream_model
from .events import check_submission, handle_submission_error


import gradio as gr
from PIL import Image


def process_caption_submission(
    model_path_selected: str,
    pil_image: Image.Image,
    caption_length: str,
    max_tokens_val: int,
    temperature_val: float,
    top_p_val: float,
):
    """
    Process query task submission.

    Args:
        model_path_selected (str): Path to the model file
        pil_image (PIL.Image): Image to query
        caption_length (str): Desired caption length
        max_tokens_val (int): Maximum tokens to generate
        temperature_val (float): Temperature for generation
        top_p_val (float): Top-p value for generation

    Returns:
        tuple: (image, task_name, prompt, response)
    """
    task_name = "Caption"
    image_col_update = gr.update(scale=1)
    text_col_update = gr.update(scale=7)

    # Check for submission errors
    error_message = check_submission(
        model_path_selected,
        pil_image,
        caption_length,
    )
    if error_message:
        return handle_submission_error(
            error_message,
            image_col_update,
            text_col_update,
        )

    try:
        # query_moondream_model will raise gr.Error on failure (model load, core query error)
        answer = caption_image(
            model_path_selected,
            pil_image,
            caption_length,
            max_tokens_val,
            temperature_val,
            top_p_val,
        )
        return (
            pil_image,
            task_name,
            caption_length,
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
