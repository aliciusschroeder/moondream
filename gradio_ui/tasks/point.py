import gradio as gr
from PIL import Image, ImageDraw

from ..core.model_loader import load_or_get_cached_model

# SMALL_CIRCLE_RADIUS = 3
# LARGE_CIRCLE_RADIUS = 20


def point_objects(
    model_path_selected: str,
    pil_image: Image.Image,
    object_label: str,
    max_objects: int,
):
    """
    Query the Moondream model with an image and object label.

    Args:
        model_path_selected (str): Path to the model file
        pil_image (PIL.Image): Image to query
        object_label (str): Object label to point out in the image
        max_objects (int): Maximum number of objects to detect

    Returns:
        list[dict[str, float]]: List of points where the object is detected
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

        print(f"Querying model for points of: '{object_label}'...")
        object_sampling_settings = {
            "max_objects": int(max_objects),
        }
        print(f"Using object sampling settings: {object_sampling_settings}")

        result_dict = model.point(
            pil_image,
            object_label,
            object_sampling_settings,
        )

        points: list[dict[str, float]] = result_dict.get("points", [])
        print(f"Model responded: '{points}'")

        SMALL_CIRCLE_RADIUS = pil_image.width / 470
        LARGE_CIRCLE_RADIUS = pil_image.width / 70

        # Draw points on the image
        for point in points:
            x = int(point["x"] * pil_image.width)
            y = int(point["y"] * pil_image.height)
            # Draw a circle at (x, y) on the image
            draw = ImageDraw.Draw(pil_image, "RGBA")
            draw.ellipse(
                (
                    x - SMALL_CIRCLE_RADIUS,
                    y - SMALL_CIRCLE_RADIUS,
                    x + SMALL_CIRCLE_RADIUS,
                    y + SMALL_CIRCLE_RADIUS,
                ),
                fill="blue",
                outline="blue",
            )
            draw.ellipse(
                (
                    x - LARGE_CIRCLE_RADIUS,
                    y - LARGE_CIRCLE_RADIUS,
                    x + LARGE_CIRCLE_RADIUS,
                    y + LARGE_CIRCLE_RADIUS,
                ),
                fill=(0, 0, 255, 64),
                outline="blue",
                width=2,
            )

        return points, pil_image

    except gr.Error:  # Re-raise gr.Error exceptions to let Gradio handle them in the UI
        raise
    except Exception as e:  # Catch other unexpected errors
        error_message = f"An unexpected error occurred during the model query: {str(e)}"
        print(f"ERROR: {error_message}")
        # Convert other exceptions to gr.Error for consistent UI error reporting
        raise gr.Error(error_message)
