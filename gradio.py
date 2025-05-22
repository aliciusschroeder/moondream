import gradio as gr
import os
import torch
from PIL import (
    Image,
)  # PIL.Image is used as a type hint in MoondreamModel and for Gradio input

# --- Moondream Imports ---
# This section assumes the Moondream package is structured and accessible.
try:
    from moondream.torch.moondream import (
        MoondreamModel,
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
        DEFAULT_MAX_OBJECTS,
    )
    from moondream.torch.config import MoondreamConfig
    from moondream.torch.weights import load_weights_into_model
except ImportError as e:
    print("ERROR: Failed to import Moondream components.")
    print(
        "Please ensure the Moondream library is correctly installed and accessible in your PYTHONPATH."
    )
    print(f"Details: {e}")
    # Define dummy classes if import fails, so Gradio interface can attempt to load.
    if "MoondreamConfig" not in globals():

        class MoondreamConfig:
            pass

    if "MoondreamModel" not in globals():

        class MoondreamModel:
            def __init__(self, config):
                self.config = config

            def query(self, image, question, stream=False, settings=None):
                return {
                    "answer": "Error: MoondreamModel not loaded due to import failure."
                }

            def to(self, device):
                pass

            def eval(self):
                pass

    if "DEFAULT_MAX_TOKENS" not in globals():
        DEFAULT_MAX_TOKENS = 512  # Fallback default
    if "DEFAULT_TEMPERATURE" not in globals():
        DEFAULT_TEMPERATURE = 0.2  # Fallback default
    if "DEFAULT_TOP_P" not in globals():
        DEFAULT_TOP_P = 1.0  # Fallback default
    if "DEFAULT_MAX_OBJECTS" not in globals():
        DEFAULT_MAX_OBJECTS = 10  # Fallback default

    if "load_weights_into_model" not in globals():

        def load_weights_into_model(weights_file, model):
            raise ImportError("load_weights_into_model is not available.")


# --- End Moondream Imports ---

# --- Configuration ---
MODEL_DIR = "/home/alec/moondream/models/"  # Directory to scan for model files

# --- Global Model Cache ---
cached_model_obj = None
cached_model_path_str = None
# --- End Global Model Cache ---


def get_model_files_from_directory(directory: str) -> list:
    """Scans the specified directory for model files."""
    if not os.path.exists(directory):
        print(
            f"Warning: Model directory '{directory}' does not exist. Please create it and add model files."
        )
        return []

    try:
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and (f.endswith(".pt") or f.endswith(".pth") or f.endswith(".safetensors"))
        ]
        if not files:
            print(
                f"Warning: No compatible model files (e.g., .pt, .pth, .safetensors) found in '{directory}'."
            )
        return files
    except Exception as e:
        print(f"Error scanning model directory '{directory}': {e}")
        return []


def load_or_get_cached_model(selected_model_path: str):
    """
    Loads the Moondream model from the given path or returns a cached version.
    Handles moving the model to CUDA (if available) or CPU.
    """
    global cached_model_obj, cached_model_path_str

    if not selected_model_path:
        raise gr.Error("No model path provided. Cannot load model.")

    if selected_model_path == cached_model_path_str and cached_model_obj is not None:
        print(f"Using cached model: {selected_model_path}")
        return cached_model_obj

    print(f"Attempting to load model from: {selected_model_path}...")
    try:
        config = MoondreamConfig()
        model = MoondreamModel(config=config)
        load_weights_into_model(weights_file=selected_model_path, model=model)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        cached_model_obj = model
        cached_model_path_str = selected_model_path

        print(
            f"Model loaded successfully from '{selected_model_path}' and moved to {device}."
        )
        return model
    except Exception as e:
        cached_model_obj = None
        cached_model_path_str = None
        error_message = f"Failed to load model from '{selected_model_path}': {str(e)}"
        print(f"ERROR: {error_message}")
        raise gr.Error(error_message)


model_files_list = get_model_files_from_directory(MODEL_DIR)

initial_model_load_status_message = "No models found or directory empty."
if model_files_list:
    selected_initial_model = model_files_list[0]
    initial_model_load_status_message = f"Attempting to load initial model: {os.path.basename(selected_initial_model)}..."
    print(initial_model_load_status_message)
    try:
        # Ensure MoondreamConfig and MoondreamModel are available before use
        # This check is important if dummy classes were defined due to import errors
        if (
            "MoondreamConfig" in globals()
            and callable(MoondreamConfig)
            and "MoondreamModel" in globals()
            and callable(MoondreamModel)
            and "load_weights_into_model" in globals()
            and callable(load_weights_into_model)
            and not isinstance(MoondreamConfig(), type(None))
            and not (
                hasattr(MoondreamModel, "query")
                and "import failure"
                in MoondreamModel(MoondreamConfig()).query(None, None)["answer"]
            )
        ):  # Check if not dummy

            cached_model_obj = load_or_get_cached_model(selected_initial_model)
            initial_model_load_status_message = (
                f"✅ Initial model '{os.path.basename(selected_initial_model)}' loaded."
            )
            print(initial_model_load_status_message)
        else:
            initial_model_load_status_message = "❌ Moondream components not fully available for initial load. Check imports and library installation."
            print(
                "ERROR: Moondream components (Config, Model, load_weights) appear to be dummy versions or not fully imported for initial load."
            )
            cached_model_obj = None
            cached_model_path_str = None

    except (
        Exception
    ) as e:  # Catch gr.Error or other exceptions from load_or_get_cached_model
        initial_model_load_status_message = f"❌ Error loading initial model '{os.path.basename(selected_initial_model)}': {e}"
        print(f"ERROR during initial model load: {e}")
        cached_model_obj = None
        cached_model_path_str = None


def query_moondream_model(
    model_path_selected: str,
    pil_image: Image.Image,
    question_text: str,
    max_tokens_val: int,
    temperature_val: float,
    top_p_val: float,
):
    # Input validation for core function (already done in wrapper, but good for direct use)
    if not model_path_selected:
        # Using gr.Warning here would show up if called directly, but it's better handled by the UI wrapper.
        # For direct calls, raising an error or returning an error status is more appropriate.
        # Since this is called by a Gradio handler, gr.Error is suitable.
        raise gr.Error(
            "No model selected for query. Please choose a model from Settings."
        )
    if pil_image is None:
        raise gr.Error("No image provided for query. Please upload an image.")
    if not question_text or not question_text.strip():
        raise gr.Error("No question provided for query. Please enter a question.")

    try:
        model = load_or_get_cached_model(model_path_selected)  # This can raise gr.Error

        # Model should not be None if load_or_get_cached_model didn't raise gr.Error
        # but defensive check is okay if there's any doubt.
        if model is None:
            raise gr.Error(
                "Model could not be loaded. Check application logs and settings."
            )

        print(f"Querying model with question: '{question_text}'...")
        text_sampling_settings = {
            "max_tokens": int(max_tokens_val),
            "temperature": float(temperature_val),
            "top_p": float(top_p_val),
        }
        print(f"Using text sampling settings: {text_sampling_settings}")

        result_dict = model.query(
            image=pil_image,
            question=question_text,
            stream=False,
            settings=text_sampling_settings,
        )

        answer = result_dict.get("answer", "No answer returned by the model.")
        print(f"Model responded: '{answer}'")
        return answer  # Return only the answer string for success

    except gr.Error:  # Re-raise gr.Error exceptions to let Gradio handle them in the UI
        raise
    except Exception as e:  # Catch other unexpected errors
        error_message = f"An unexpected error occurred during the model query: {str(e)}"
        print(f"ERROR: {error_message}")
        # Convert other exceptions to gr.Error for consistent UI error reporting
        raise gr.Error(error_message)


def handle_model_selection_change(selected_model_path: str):
    if not selected_model_path:
        yield "⚠️ No model selected. Please choose a model."
        return

    yield f"⏳ Loading model: {os.path.basename(selected_model_path)}..."
    try:
        load_or_get_cached_model(
            selected_model_path
        )  # This function raises gr.Error on failure
        yield f"✅ Model '{os.path.basename(selected_model_path)}' loaded successfully."
    except gr.Error as ge:  # Catch gr.Error from load_or_get_cached_model
        yield f"❌ Error loading model: {str(ge)}"  # gr.Error's string representation is user-friendly
    except (
        Exception
    ) as e:  # Catch any other unexpected errors during the loading process
        yield f"❌ Failed to load model '{os.path.basename(selected_model_path)}': An unexpected error occurred: {str(e)}"


# --- Gradio Interface Definition ---
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky
    )
) as demo:
    gr.Markdown(
        f"""
        # 🌔 Moondream Interface (Refactored from Sketch)
        Upload an image, configure settings, choose a task, provide input, and see the results.
        Models are loaded from: `{MODEL_DIR}`.
        """
    )

    with gr.Row():  # Main layout: Left column for Image & Settings, Right for Tasks & Results
        # --- Left Column (scale=1) ---
        with gr.Column(scale=1):
            # Top-Left: IMAGE (Sketch)
            main_image_uploader = gr.Image(type="pil", label="IMAGE (Upload Input)")

            # Bottom-Left: SETTINGS (Sketch)
            with gr.Accordion("SETTINGS", open=True):  # Main accordion for all settings
                with gr.Accordion(
                    "General", open=True
                ):  # Collapsible sub-section for general settings
                    model_path_dropdown = gr.Dropdown(
                        label="Select Model File",
                        choices=model_files_list,
                        value=model_files_list[0] if model_files_list else None,
                        info=f"Models from {MODEL_DIR}",
                    )
                    model_load_status_md = gr.Markdown(
                        initial_model_load_status_message
                    )

                with gr.Accordion(
                    "Text Generation Settings", open=False
                ):  # Collapsible sub-section
                    max_tokens_slider = gr.Slider(
                        minimum=1,
                        maximum=2048,
                        value=DEFAULT_MAX_TOKENS,
                        step=1,
                        label="Max Tokens",
                        info="Max tokens for text generation.",
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=DEFAULT_TEMPERATURE,
                        step=0.01,
                        label="Temperature",
                        info="Randomness control. Lower is more deterministic.",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=DEFAULT_TOP_P,
                        step=0.01,
                        label="Top P",
                        info="Nucleus sampling parameter.",
                    )

                with gr.Accordion(
                    "Object Settings", open=False
                ):  # Collapsible sub-section
                    max_objects_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=DEFAULT_MAX_OBJECTS,
                        step=1,
                        label="Max Objects",
                        info="Max objects for detection/pointing tasks.",
                    )

        # --- Right Column (scale=2) ---
        with gr.Column(scale=2):
            # Top-Right: Tabs for tasks (Sketch: query | Caption | ...)
            with gr.Tabs() as tabs:
                with gr.TabItem("❓ Query", id="query_tab"):
                    question_textbox_query = gr.Textbox(
                        label="Query",  # Sketch label
                        placeholder="e.g., What is in this image? Describe the scene.",
                        lines=3,
                    )
                    submit_button_query = gr.Button(
                        "SUBMIT", variant="primary"
                    )  # Sketch label

                with gr.TabItem("📝 Caption", id="caption_tab"):
                    gr.Markdown(
                        "Caption generation. Settings from 'Text Generation Settings' will apply."
                    )
                    submit_button_caption = gr.Button(
                        "SUBMIT Caption (Coming Soon)", variant="secondary"
                    )
                    gr.Markdown("*Caption functionality is under development.*")

                with gr.TabItem("📍 Point", id="point_tab"):
                    gr.Markdown(
                        "Point to object. Settings from 'Object Settings' will apply."
                    )
                    object_textbox_point = gr.Textbox(
                        label="Object to Point At", placeholder="e.g., the red car"
                    )
                    submit_button_point = gr.Button(
                        "SUBMIT Point (Coming Soon)", variant="secondary"
                    )
                    gr.Markdown("*Point functionality is under development.*")

                with gr.TabItem("👁️ Detect", id="detect_tab"):
                    gr.Markdown(
                        "Detect objects. Settings from 'Object Settings' will apply."
                    )
                    object_textbox_detect = gr.Textbox(
                        label="Object to Detect", placeholder="e.g., cat, table"
                    )
                    submit_button_detect = gr.Button(
                        "SUBMIT Detect (Coming Soon)", variant="secondary"
                    )
                    gr.Markdown("*Detect functionality is under development.*")

            # Bottom-Center/Right: RESULT area (Sketch: IMAGE | TASK | RESULT TEXT)
            gr.Markdown("---")
            gr.Markdown("## RESULT")
            with gr.Group():
                result_image_display = gr.Image(
                    label="Input Image (Context)", interactive=False
                )
                result_task_display = gr.Textbox(label="Chosen Task", interactive=False)
                result_prompt_display = gr.Textbox(
                    label="User Prompt / Question", interactive=False
                )
                result_text_output = gr.Textbox(
                    label="Model Response (Result Text)", interactive=False, lines=10
                )

    # --- Event Handlers ---
    model_path_dropdown.change(
        fn=handle_model_selection_change,
        inputs=[model_path_dropdown],
        outputs=[model_load_status_md],
    )

    def process_query_submission(
        model_path_selected: str,
        pil_image: Image.Image,
        question_text: str,
        max_tokens_val: int,
        temperature_val: float,
        top_p_val: float,
    ):
        task_name = "Query"
        # UI-level validation: Use gr.Warning for recoverable issues, actual call raises gr.Error
        if pil_image is None:
            gr.Warning("Please upload an image for the query.")
            # Return values to clear/indicate error in result fields
            return (
                None,
                task_name,
                question_text or "N/A",
                "Error: Image required for query. Please upload an image.",
            )
        if not question_text or not question_text.strip():
            gr.Warning("Please enter a question for the query.")
            return (
                pil_image,
                task_name,
                "No question provided.",
                "Error: Question required for query. Please enter a question.",
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
            return pil_image, task_name, question_text, answer

        except gr.Error as ge:
            # Gradio will display this error automatically.
            # We still return values to update the result fields appropriately, showing the error message.
            print(f"Gradio Error caught in process_query_submission: {ge}")
            return pil_image, task_name, question_text, f"Operation Failed: {str(ge)}"
        except Exception as e:
            # Catch any other unexpected errors in this wrapper that weren't converted to gr.Error
            error_msg = (
                f"An unexpected error occurred during query processing: {str(e)}"
            )
            print(f"ERROR: {error_msg}")
            # Optionally, re-raise as gr.Error or return error message
            return pil_image, task_name, question_text, error_msg

    submit_button_query.click(
        fn=process_query_submission,
        inputs=[
            model_path_dropdown,
            main_image_uploader,
            question_textbox_query,
            max_tokens_slider,
            temperature_slider,
            top_p_slider,
        ],
        outputs=[
            result_image_display,
            result_task_display,
            result_prompt_display,
            result_text_output,
        ],
        api_name="query_image",
    )

    # Placeholder handlers for other tasks
    def placeholder_task_handler(
        model_path, image, task_input_value, task_name_str, relevant_settings_dict
    ):
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

    submit_button_caption.click(
        fn=lambda model, img, mt, temp, tp: placeholder_task_handler(
            model,
            img,
            "N/A (No direct text input for caption)",
            "Caption",
            {"max_tokens": mt, "temp": temp, "top_p": tp},
        ),
        inputs=[
            model_path_dropdown,
            main_image_uploader,
            max_tokens_slider,
            temperature_slider,
            top_p_slider,
        ],
        outputs=[
            result_image_display,
            result_task_display,
            result_prompt_display,
            result_text_output,
        ],
    )

    submit_button_point.click(
        fn=lambda model, img, obj_text, mo: placeholder_task_handler(
            model, img, obj_text, "Point", {"max_objects": mo}
        ),
        inputs=[
            model_path_dropdown,
            main_image_uploader,
            object_textbox_point,
            max_objects_slider,
        ],
        outputs=[
            result_image_display,
            result_task_display,
            result_prompt_display,
            result_text_output,
        ],
    )

    submit_button_detect.click(
        fn=lambda model, img, obj_text, mo: placeholder_task_handler(
            model, img, obj_text, "Detect", {"max_objects": mo}
        ),
        inputs=[
            model_path_dropdown,
            main_image_uploader,
            object_textbox_detect,
            max_objects_slider,
        ],
        outputs=[
            result_image_display,
            result_task_display,
            result_prompt_display,
            result_text_output,
        ],
    )

    gr.Markdown(
        """
        ---
        **Notes & Sketch Alignment:**
        - **IMAGE (Top-Left):** Main image uploader.
        - **SETTINGS (Bottom-Left):** Collapsible sections for General, Text Gen, Objects.
        - **query | Caption | ... (Top-Right):** Implemented as Tabs. Each tab has relevant inputs and a SUBMIT button.
        - **RESULT (Bottom-Right):** Displays:
            - Input Image (Context)
            - Chosen Task
            - User Prompt / Question
            - Model Response (Result Text)
        - Model loading status and errors are reported. Check console for detailed logs.
        """
    )

if __name__ == "__main__":
    print("Starting Moondream Gradio Interface (Sketch-Refactored)...")
    if not model_files_list:
        print(
            f"CRITICAL WARNING: No model files were found in the specified directory: {MODEL_DIR}."
        )

    demo.launch(debug=True)
