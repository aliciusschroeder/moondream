import gradio as gr
from PIL import Image, ImageDraw

from ..core.model_loader import load_or_get_cached_model


def detect_objects(
    model_path_selected: str,
    pil_image: Image.Image,
    object_label: str,
    max_objects: int,
    return_just_points: bool = False,
):
    """
    Query the Moondream model with an image and object label.

    Args:
        model_path_selected (str): Path to the model file
        pil_image (PIL.Image): Image to query
        object_label (str): Object label to point out in the image
        max_objects (int): Maximum number of objects to detect

    Returns:
        TODO: clarify return types
        PIL.Image: Image with points drawn on it

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
    if not object_label or not object_label.strip():
        raise gr.Error(
            "No object label provided for query. Please enter an object label."
        )

    try:
        model = load_or_get_cached_model(model_path_selected)
        if model is None:
            raise gr.Error(
                "Model could not be loaded. Check application logs and settings."
            )

        print(f"Querying model for bounding boxes of: '{object_label}'...")
        object_sampling_settings = {
            "max_objects": int(max_objects),
        }
        print(f"Using object sampling settings: {object_sampling_settings}")

        result_dict = model.detect(
            pil_image,
            object_label,
            object_sampling_settings,
        )

        objects: list[dict[str, float]] = result_dict.get("objects", [])
        print(f"Model responded: '{objects}'")

        if return_just_points:
            return objects, None

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(pil_image)
        bounding_box_width = max(pil_image.width // 700, 1)
        text_margin_left = pil_image.width / 470
        for obj in objects:
            x1, y1, x2, y2 = (
                obj["x_min"] * pil_image.width,
                obj["y_min"] * pil_image.height,
                obj["x_max"] * pil_image.width,
                obj["y_max"] * pil_image.height,
            )
            draw.rectangle([x1, y1, x2, y2], outline="red", width=bounding_box_width)
            # Calculate font size based on image size and bounding box size
            font_size = min(max(int((x2 - x1) / 10), 6), 64)

            draw.text((x1 + text_margin_left, y1), object_label, fill="red", font_size=font_size)

        return objects, pil_image

    except gr.Error:  # Re-raise gr.Error exceptions to let Gradio handle them in the UI
        raise
    except Exception as e:  # Catch other unexpected errors
        error_message = f"An unexpected error occurred during the model query: {str(e)}"
        print(f"ERROR: {error_message}")
        # Convert other exceptions to gr.Error for consistent UI error reporting
        raise gr.Error(error_message)
