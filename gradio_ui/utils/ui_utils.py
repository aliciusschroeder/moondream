import os


def create_model_choices(model_paths: list[str]) -> list[tuple[str, str]]:
    """
    Create a list of tuples for model choices in the Gradio interface.

    Args:
        model_paths (list[str]): List of model paths.

    Returns:
        list[tuple[str, str]]: List of tuples containing model name and path.
    """
    return [(os.path.basename(path), path) for path in model_paths]
