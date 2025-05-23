import json
import random
import gradio as gr
from PIL import Image, ImageDraw

from .detect import detect_objects
from ..core.model_loader import load_or_get_cached_model
from moondream.torch.image_crops import select_tiling

CROP_SIZE = 378
DEFAULT_MAX_TILES = 12


def split_image_into_tiles(
    image: Image.Image, max_tiles: int = DEFAULT_MAX_TILES
) -> list[Image.Image]:
    h_tiles, w_tiles = select_tiling(image.height, image.width, CROP_SIZE, max_tiles)
    print(f"Splitting image into {w_tiles}x{h_tiles} (WxH) tiles.")

    width, height = image.size
    tile_width = width // w_tiles
    tile_height = height // h_tiles

    # Adjust the step to slightly overlap if not evenly divisible
    step_x = (width - tile_width) // (w_tiles - 1) if w_tiles > 1 else 0
    step_y = (height - tile_height) // (h_tiles - 1) if h_tiles > 1 else 0

    tiles = []
    for i in range(h_tiles):
        for j in range(w_tiles):
            left = min(j * step_x, width - tile_width)
            upper = min(i * step_y, height - tile_height)
            right = left + tile_width
            lower = upper + tile_height
            tile = image.crop((left, upper, right, lower))
            tiles.append(tile)

    return tiles


def detect_all_objects(
    model_path_selected: str,
    pil_image: Image.Image,
    max_objects: int,
    in_depth: bool = True,
    max_tiles: int = DEFAULT_MAX_TILES,
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

        objects = []
        tiles = [pil_image]
        if in_depth:
            tiles.extend(split_image_into_tiles(pil_image, max_tiles))

        # Process each tile separately
        for tile in tiles:
            # Ensure the tile is in RGB mode
            if tile.mode != "RGB":
                tile = tile.convert("RGB")
            # Resize the tile to a fixed size
            tile = tile.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)

            # Process the tile with the model
            result_dict = model.query(
                image=tile,
                question=objects_query,
                stream=False,
                settings=text_sampling_settings,
            )

            answer = result_dict.get("answer", "[]")
            print(f"Model found objects: '{answer}'")

            # Parse JSON response, handle potential formatting issues
            try:
                content = json.loads(answer)
                if isinstance(content, list):
                    objects.extend(
                        [
                            item.strip().lower()
                            for item in content
                            if isinstance(item, str)
                        ]
                    )
                else:
                    print("Answer was parsed but not a list:", answer)
                continue
            except json.JSONDecodeError:
                # If the response is not a valid JSON, handle it gracefully
                print(f"Failed to parse JSON response: {answer}")
                continue
            except gr.Error as ge:
                print("Error during object detection:", str(ge))
                continue
        full_count = len(objects)
        # Remove duplicates
        objects = list(set(objects))
        print(f"Found {full_count} objects, reduced to {len(objects)} unique objects:")
        print("\n".join(objects))

        try:
            # Try to parse the full response as JSON
            detections = []
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
