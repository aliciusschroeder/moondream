"""
Placeholder tasks for features that are under development.
"""

import gradio as gr


def placeholder_task_handler(
    model_path, image, task_input_value, task_name_str, relevant_settings_dict
):
    """
    Generic placeholder handler for tasks not yet implemented.

    Args:
        model_path (str): Path to the model file
        image (PIL.Image): Image for the task
        task_input_value (str): Task-specific input
        task_name_str (str): Name of the task
        relevant_settings_dict (dict): Task-specific settings

    Returns:
        tuple: (image, task_name, input_display, response)
    """
    # Basic UI validation for image (common to most tasks)
    if image is None:
        gr.Warning(f"Please upload an image for the {task_name_str} task.")
        return (
            None,
            task_name_str,
            str(task_input_value) if task_input_value else "N/A",
            f"Error: Image required for {task_name_str}.",
        )

    # Specific input validation (example)
    if task_name_str in ["Point", "Detect"] and (
        not task_input_value or not str(task_input_value).strip()
    ):
        gr.Warning(f"Please provide an object name for the {task_name_str} task.")
        return (
            image,
            task_name_str,
            "No object specified.",
            f"Error: Object name required for {task_name_str}.",
        )

    response_message = f"{task_name_str} task is not yet implemented. "
    response_message += (
        f"Input: '{task_input_value}'. Settings: {relevant_settings_dict}"
    )
    return (
        image,
        task_name_str,
        str(task_input_value) if task_input_value else "N/A",
        response_message,
    )
