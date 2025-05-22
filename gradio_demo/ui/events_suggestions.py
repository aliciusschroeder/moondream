from ..tasks.suggestions import get_question_suggestions
from ..utils.img_utils import get_image_hash


import gradio as gr


def question_suggestions_event(
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
