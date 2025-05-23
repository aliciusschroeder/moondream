from ..tasks.suggestions import get_question_suggestions, get_object_suggestions
from ..utils.img_utils import get_image_hash
from PIL import Image


import gradio as gr


def meta_suggestions_handler(
    model_path_selected: str,
    image: Image.Image,
    last_processed_image: dict[str, str | None],
    current_tab: str,
    target_tab: str,
    generation_fn: callable,
):
    """
    Generate suggestions for the current tab and image.

    Args:
        model_path_selected (str): Path to the selected model
        image (PIL.Image): The current image
        last_processed_image (dict[str, str | None]): Hashes of the last processed images
        current_tab (str): The current tab ID
        target_tab (str): The target tab ID to check against
        generation_fn (callable): Function to generate suggestions

    Returns:
        tuple: Updated UI elements and state
    """
    # Check if we need to generate new suggestions
    print(f"Current tab: {current_tab}, Target tab: {target_tab}")
    if current_tab != target_tab or image is None:
        visibility = gr.update(visible=False) if image is None else gr.update()
        return (
            gr.update(),
            visibility,
            visibility,
            visibility,
            last_processed_image,
        )

    # Get hash of current image
    current_hash = get_image_hash(image)

    # Check if image changed or first time
    if current_hash == last_processed_image[target_tab]:
        # Image hasn't changed, keep existing suggestions
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            last_processed_image,
        )

    # Image changed or first time, generate new suggestions
    try:
        # Get suggestions from model
        suggestions = generation_fn(model_path_selected, image)
        new_hashes = last_processed_image.copy()
        new_hashes[target_tab] = current_hash

        # Update buttons with suggestions
        return (
            gr.update(value="", visible=False),  # Hide status
            gr.update(value=suggestions[0], visible=True),
            gr.update(value=suggestions[1], visible=True),
            gr.update(value=suggestions[2], visible=True),
            new_hashes,  # Update processed image hashes
        )
    except gr.Error as e:
        # Error occurred, show message and hide buttons
        return (
            gr.update(value=f"*{str(e)}*", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            last_processed_image,  # Keep old hashes
        )
    except Exception as e:
        # Unexpected error, show message and hide buttons
        error_msg = f"*Error generating suggestions: {str(e)}*"
        return (
            gr.update(value=error_msg, visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            last_processed_image,  # Keep old hashes
        )


def question_suggestions_event(
    model_path_selected, image, last_processed_image, current_tab
):
    return meta_suggestions_handler(
        model_path_selected,
        image,
        last_processed_image,
        current_tab,
        "query_tab",
        get_question_suggestions,
    )


def object_suggestions_event(
    model_path_selected, image, last_processed_image, current_tab, target_tab
):
    return meta_suggestions_handler(
        model_path_selected,
        image,
        last_processed_image,
        current_tab,
        target_tab,
        get_object_suggestions,
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
