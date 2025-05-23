import json
import random
import gradio as gr
from PIL import Image, ImageDraw

from .detect import detect_objects
from ..core.model_loader import load_or_get_cached_model


def detect_all_objects(
    model_path_selected: str,
    pil_image: Image.Image,
    max_objects: int,
):
    """
    Query the Moondream model with an image and object label.

    Args:
        model_path_selected (str): Path to the model file
        pil_image (PIL.Image): Image to query
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

    try:
        model = load_or_get_cached_model(model_path_selected)
        if model is None:
            raise gr.Error(
                "Model could not be loaded. Check application logs and settings."
            )

        print("Generating object list for image...")
        objects_query = "Return a json list of all objects in this image"

        text_sampling_settings = {
            "max_tokens": 1900,
            "temperature": 0.5,
            "top_p": 0.3,
        }

        result_dict = model.query(
            image=pil_image,
            question=objects_query,
            stream=False,
            settings=text_sampling_settings,
        )

        answer = result_dict.get("answer", "[]")
        print(f"Model found objects: '{answer}'")

        # Parse JSON response, handle potential formatting issues
        try:
            # Try to parse the full response as JSON
            objects = json.loads(answer)
            detections = []
            if isinstance(objects, list):
                for obj in objects:
                    try:
                        points, _ = detect_objects(
                            model_path_selected,
                            pil_image,
                            obj,
                            max_objects,
                            return_just_points=True,
                        )
                        for point in points:
                            detections.append((obj, point))
                            print("Detected object:", obj, "at point:", point)
                    except gr.Error as ge:
                        print("Error during object detection:", str(ge))
                        continue
                    except Exception as e:
                        print("Unexpected error during object detection:", str(e))
                        continue

                print("Final detections:", detections)

                draw = ImageDraw.Draw(pil_image)
                bounding_box_width = max(pil_image.width // 700, 1)
                text_margin_left = pil_image.width / 470
                past_detection = ""
                current_color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                for obj, point in detections:
                    if obj != past_detection:
                        current_color = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        )
                        past_detection = obj
                    x1, y1, x2, y2 = (
                        point["x_min"] * pil_image.width,
                        point["y_min"] * pil_image.height,
                        point["x_max"] * pil_image.width,
                        point["y_max"] * pil_image.height,
                    )
                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline=current_color,
                        width=bounding_box_width,
                    )
                    font_size = min(max(int((x2 - x1) / 10), 6), 64)
                    draw.text(
                        (x1 + text_margin_left, y1),
                        obj,
                        fill=current_color,
                        font_size=font_size,
                    )
            else:
                print("Answer was parsed but not a list:", answer)
        except json.JSONDecodeError:
            # If the response is not a valid JSON, handle it gracefully
            print(f"ERROR: Failed to parse JSON response: {answer}")
            raise gr.Error(
                "Failed to parse the model's response. Please check the model output."
            )

        return detections, pil_image

    except gr.Error:  # Re-raise gr.Error exceptions to let Gradio handle them in the UI
        raise
    except Exception as e:  # Catch other unexpected errors
        error_message = f"An unexpected error occurred during the model query: {str(e)}"
        print(f"ERROR: {error_message}")
        # Convert other exceptions to gr.Error for consistent UI error reporting
        raise gr.Error(error_message)
