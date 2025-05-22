"""
Defines the full Gradio UI layout and returns the interface demo.
"""

import gradio as gr

from ..utils.ui_utils import create_model_choices
from ..core.config import APP_TITLE, APP_DESCRIPTION, PRIMARY_HUE, SECONDARY_HUE
from ..moondream_imports import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_OBJECTS,
)


def create_gradio_ui(model_files_list, initial_model_status):
    """
    Create and configure the Gradio UI.

    Args:
        model_files_list (list): List of model file paths
        initial_model_status (str): Initial model loading status message

    Returns:
        gr.Blocks: The configured Gradio interface
    """
    from .events import handle_model_selection_change, process_query_submission
    from ..tasks import placeholder_task_handler

    # Create the Gradio Blocks interface
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue=PRIMARY_HUE, secondary_hue=SECONDARY_HUE)
    ) as demo:
        gr.Markdown(APP_DESCRIPTION)

        with gr.Row():  # Main layout: Left column for Image & Settings, Right for Tasks & Results
            # --- Left Column (scale=1) ---
            with gr.Column(scale=1):
                # Top-Left: IMAGE (Sketch)
                main_image_uploader = gr.Image(type="pil", label="IMAGE (Upload Input)")

                # Bottom-Left: SETTINGS (Sketch)
                with gr.Accordion(
                    "SETTINGS", open=True
                ):  # Main accordion for all settings
                    with gr.Accordion(
                        "General", open=True
                    ):  # Collapsible sub-section for general settings
                        model_path_dropdown = gr.Dropdown(
                            label="Select Model File",
                            choices=create_model_choices(model_files_list),
                            value=model_files_list[0] if model_files_list else None,
                            info="Select a model file to load",
                        )
                        model_load_status_md = gr.Markdown(initial_model_status)

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
                            placeholder="Enter a visual query, e.g. What is in this image? Describe the scene.",
                            value="Describe the image.",
                            lines=3,
                        )
                        submit_button_query = gr.Button(
                            "SUBMIT", variant="primary"
                        )  # Sketch label

                    with gr.TabItem("📝 Caption", id="caption_tab"):
                        gr.Markdown(
                            "Caption generation. Settings from 'Text Generation Settings' will apply."
                        )
                        caption_length_selector = gr.Dropdown(
                            label="Caption Length",
                            choices=["Short", "Medium", "Long"],
                            value="Medium",
                            info="Select the desired length of the caption.",
                        )
                        submit_button_caption = gr.Button(
                            "SUBMIT", variant="primary"
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
                            "SUBMIT", variant="primary"
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
                            "SUBMIT", variant="primary"
                        )
                        gr.Markdown("*Detect functionality is under development.*")

                # Bottom-Center/Right: RESULT area (Sketch: IMAGE | TASK | RESULT TEXT)
                gr.Markdown("---")
                gr.Markdown("## RESULT")
                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=1):
                            result_image_display = gr.Image(
                                label="Input Image (Context)", interactive=False
                            )
                            result_task_display = gr.Textbox(
                                label="Chosen Task", interactive=False
                            )
                            result_prompt_display = gr.Textbox(
                                label="User Prompt / Question",
                                interactive=False,
                                show_copy_button=True,
                            )
                        with gr.Column(scale=2):
                            result_text_output = gr.Textbox(
                                label="Model Response (Result Text)",
                                interactive=False,
                                lines=10,
                                show_copy_button=True,
                            )

        # --- Event Handlers ---
        model_path_dropdown.change(
            fn=handle_model_selection_change,
            inputs=[model_path_dropdown],
            outputs=[model_load_status_md],
        )

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

        # Caption task handler
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

        # Point task handler
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

        # Detect task handler
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
    return demo
