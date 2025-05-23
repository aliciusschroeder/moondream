from gradio_demo.core.model_loader import load_or_get_cached_model


import gradio as gr
from PIL import Image


import json


def get_question_suggestions(model_path_selected: str, pil_image: Image.Image):
    """
    Get question suggestions for an image from the Moondream model.

    Args:
        model_path_selected (str): Path to the model file
        pil_image (PIL.Image): Image to generate questions for

    Returns:
        list: List of 3 suggested questions

    Raises:
        gr.Error: If inputs are invalid or query fails
    """
    # Input validation
    if not model_path_selected:
        raise gr.Error("No model selected. Please choose a model from Settings.")
    if pil_image is None:
        raise gr.Error("No image provided. Please upload an image.")

    try:
        model = load_or_get_cached_model(model_path_selected)

        if model is None:
            raise gr.Error(
                "Model could not be loaded. Check application logs and settings."
            )

        print("Generating question suggestions for image...")
        suggestion_query = (
            "Return a json list of 3 questions about this image's content"
        )

        text_sampling_settings = {
            "max_tokens": 150,
            "temperature": 0.5,
            "top_p": 0.3,
        }

        result_dict = model.query(
            image=pil_image,
            question=suggestion_query,
            stream=False,
            settings=text_sampling_settings,
        )

        answer = result_dict.get("answer", "[]")
        print(f"Model suggested questions: '{answer}'")

        # Parse JSON response, handle potential formatting issues
        try:
            # Try to parse the full response as JSON
            questions = json.loads(answer)
            if isinstance(questions, list):
                # Ensure we have exactly 3 questions
                if len(questions) > 3:
                    questions = questions[:3]
                elif len(questions) < 3:
                    # Pad with default questions if needed
                    defaults = [
                        "What's in this image?",
                        "Describe this scene in detail.",
                        "What objects do you see in the image?",
                    ]
                    questions.extend(defaults[len(questions) : 3])
                return questions
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract questions from text
            print("Failed to parse JSON response, trying text extraction")
            import re

            questions = re.findall(r'["\'](.*?)["\']', answer)
            if questions and len(questions) >= 3:
                return questions[:3]

        # Fallback to default questions
        return [
            "What's in this image?",
            "Describe this scene in detail.",
            "What objects do you see in the image?",
        ]

    except gr.Error:  # Re-raise gr.Error exceptions
        raise
    except Exception as e:
        error_message = (
            f"An error occurred while generating question suggestions: {str(e)}"
        )
        print(f"ERROR: {error_message}")
        raise gr.Error(error_message)


def get_object_suggestions(model_path_selected: str, pil_image: Image.Image):
    """
    Get object suggestions for an image from the Moondream model.

    Args:
        model_path_selected (str): Path to the model file
        pil_image (PIL.Image): Image to generate objects for

    Returns:
        list: List of 3 suggested objects

    Raises:
        gr.Error: If inputs are invalid or query fails
    """
    # Input validation
    if not model_path_selected:
        raise gr.Error("No model selected. Please choose a model from Settings.")
    if pil_image is None:
        raise gr.Error("No image provided. Please upload an image.")

    try:
        model = load_or_get_cached_model(model_path_selected)

        if model is None:
            raise gr.Error(
                "Model could not be loaded. Check application logs and settings."
            )

        print("Generating object suggestions for image...")
        suggestion_query = "Return a json list of 3 objects in this image"

        text_sampling_settings = {
            "max_tokens": 45,
            "temperature": 0.5,
            "top_p": 0.3,
        }

        result_dict = model.query(
            image=pil_image,
            question=suggestion_query,
            stream=False,
            settings=text_sampling_settings,
        )

        answer = result_dict.get("answer", "[]")
        print(f"Model suggested objects: '{answer}'")

        # Parse JSON response, handle potential formatting issues
        try:
            # Try to parse the full response as JSON
            objects = json.loads(answer)
            if isinstance(objects, list):
                # Ensure we have exactly 3 objects
                if len(objects) > 3:
                    objects = objects[:3]
                elif len(objects) < 3:
                    # Pad with default objects if needed
                    defaults = [
                        {"object": "cat", "bbox": [0, 0, 1, 1]},
                        {"object": "dog", "bbox": [0, 0, 1, 1]},
                        {"object": "car", "bbox": [0, 0, 1, 1]},
                    ]
                    objects.extend(defaults[len(objects) : 3])
                return objects
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract objects from text
            print("Failed to parse JSON response, trying text extraction")
            import re

            objects = re.findall(r'["\'](.*?)["\']', answer)
            if objects and len(objects) >= 3:
                return objects[:3]
        # Fallback to default objects
        return [
            {"object": "cat", "bbox": [0, 0, 1, 1]},
            {"object": "dog", "bbox": [0, 0, 1, 1]},
            {"object": "car", "bbox": [0, 0, 1, 1]},
        ]
    except gr.Error:  # Re-raise gr.Error exceptions
        raise
    except Exception as e:
        error_message = (
            f"An error occurred while generating object suggestions: {str(e)}"
        )
        print(f"ERROR: {error_message}")
        raise gr.Error(error_message)
