"""
Defines the full Gradio UI layout and returns the interface demo.
"""

import gradio as gr

from .events_suggestions import question_suggestions_event, handle_suggestion_click
from .events_tasks import process_caption_submission, process_query_submission
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
    from .events import (
        handle_model_selection_change,
    )
    from ..tasks import placeholder_task_handler

    # Create the Gradio Blocks interface
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue=PRIMARY_HUE, secondary_hue=SECONDARY_HUE)
    ) as demo:
        gr.Markdown(APP_DESCRIPTION)

        # Hidden state variables for tracking
        last_processed_image = gr.State(None)  # Store hash of last processed image
        current_tab = gr.State("query_tab")  # Track current tab

        with gr.Row():  # Main layout: Left column for Image & Settings, Right for Tasks & Results
            # --- Left Column (scale=1) ---
            with gr.Column(scale=1):
                # Top-Left: IMAGE (Sketch)
                main_image_uploader = gr.Image(type="pil", label="IMAGE (Upload Input)")

            # --- Right Column (scale=2) ---
            with gr.Column(scale=2):
                # Top-Right: Tabs for tasks (Sketch: query | Caption | ...)
                with gr.Tabs() as tabs:
                    with gr.TabItem("❓ Query", id="query_tab") as query_tab:
                        question_textbox_query = gr.Textbox(
                            label="Query",
                            placeholder="Enter a visual query, e.g. What is in this image? Describe the scene.",
                            value="Describe the image.",
                            lines=3,
                        )

                        # Add suggestion buttons row
                        with gr.Row() as suggestion_row:
                            suggestion_status = gr.Markdown(
                                "*Upload an image to get question suggestions*",
                                visible=True,
                            )

                        with gr.Row() as question_buttons_row:
                            query_suggestion_btn1 = gr.Button(
                                "Suggestion 1", variant="secondary", visible=False
                            )
                            query_suggestion_btn2 = gr.Button(
                                "Suggestion 2", variant="secondary", visible=False
                            )
                            query_suggestion_btn3 = gr.Button(
                                "Suggestion 3", variant="secondary", visible=False
                            )

                        submit_button_query = gr.Button("SUBMIT", variant="primary")

                    with gr.TabItem("📝 Caption", id="caption_tab") as caption_tab:
                        gr.Markdown(
                            "Caption generation. Settings from 'Text Generation Settings' will apply."
                        )
                        caption_length_selector = gr.Dropdown(
                            label="Caption Length",
                            choices=[
                                ("Short", "short"),
                                ("Medium", "normal"),
                                ("Long", "long"),
                            ],
                            value="normal",
                            interactive=True,
                            info="Select the desired length of the caption.",
                        )
                        submit_button_caption = gr.Button("SUBMIT", variant="primary")

                    with gr.TabItem("📍 Point", id="point_tab") as point_tab:
                        gr.Markdown(
                            "Point to object. Settings from 'Object Settings' will apply."
                        )
                        object_textbox_point = gr.Textbox(
                            label="Object to Point At", placeholder="e.g., the red car"
                        )
                        submit_button_point = gr.Button("SUBMIT", variant="primary")
                        gr.Markdown("*Point functionality is under development.*")

                    with gr.TabItem("👁️ Detect", id="detect_tab") as detect_tab:
                        gr.Markdown(
                            "Detect objects. Settings from 'Object Settings' will apply."
                        )
                        object_textbox_detect = gr.Textbox(
                            label="Object to Detect", placeholder="e.g., cat, table"
                        )
                        submit_button_detect = gr.Button("SUBMIT", variant="primary")
                        gr.Markdown("*Detect functionality is under development.*")

                # Bottom-Center/Right: RESULT area (Sketch: IMAGE | TASK | RESULT TEXT)

        with gr.Row():
            gr.Markdown("---")
        with gr.Row():
            gr.Markdown("## RESULT")

        with gr.Row():
            with gr.Column(scale=1) as result_image_col:
                result_image_display = gr.Image(
                    label="Input Image (Context)", interactive=False
                )
                result_task_display = gr.Textbox(label="Chosen Task", interactive=False)
                result_prompt_display = gr.Textbox(
                    label="User Prompt / Question",
                    interactive=False,
                    show_copy_button=True,
                )
            with gr.Column(scale=2) as result_text_col:
                result_text_output = gr.Textbox(
                    label="Model Response (Result Text)",
                    interactive=False,
                    lines=10,
                    show_copy_button=True,
                )

        with gr.Sidebar():
            # with gr.Accordion(
            #         "SETTINGS", open=True
            #     ):  # Main accordion for all settings
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

            with gr.Accordion("Object Settings", open=False):  # Collapsible sub-section
                max_objects_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=DEFAULT_MAX_OBJECTS,
                    step=1,
                    label="Max Objects",
                    info="Max objects for detection/pointing tasks.",
                )

        # --- Event Handlers ---
        model_path_dropdown.change(
            fn=handle_model_selection_change,
            inputs=[model_path_dropdown],
            outputs=[model_load_status_md],
        )

        # Track tab changes
        def update_current_tab():
            print(f"Current tab: {current_tab.value}")
            return current_tab.value

        query_tab.select(
            lambda: "query_tab",
            inputs=None,
            outputs=[current_tab],
        )
        caption_tab.select(
            lambda: "caption_tab",
            inputs=None,
            outputs=[current_tab],
        )
        point_tab.select(
            lambda: "point_tab",
            inputs=None,
            outputs=[current_tab],
        )
        detect_tab.select(
            lambda: "detect_tab",
            inputs=None,
            outputs=[current_tab],
        )

        current_tab.change(
            fn=question_suggestions_event,
            inputs=[
                model_path_dropdown,
                main_image_uploader,
                last_processed_image,
                current_tab,
            ],
            outputs=[
                suggestion_status,
                query_suggestion_btn1,
                query_suggestion_btn2,
                query_suggestion_btn3,
                question_textbox_query,
                last_processed_image,
            ],
        )

        # Generate suggestions when image is uploaded and query tab is active
        main_image_uploader.change(
            fn=question_suggestions_event,
            inputs=[
                model_path_dropdown,
                main_image_uploader,
                last_processed_image,
                current_tab,
            ],
            outputs=[
                suggestion_status,
                query_suggestion_btn1,
                query_suggestion_btn2,
                query_suggestion_btn3,
                question_textbox_query,
                last_processed_image,
            ],
        )

        # Make suggestion buttons update the query input when clicked
        query_suggestion_btn1.click(
            fn=handle_suggestion_click,
            inputs=[query_suggestion_btn1],
            outputs=[question_textbox_query],
        )

        query_suggestion_btn2.click(
            fn=handle_suggestion_click,
            inputs=[query_suggestion_btn2],
            outputs=[question_textbox_query],
        )

        query_suggestion_btn3.click(
            fn=handle_suggestion_click,
            inputs=[query_suggestion_btn3],
            outputs=[question_textbox_query],
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
                result_image_col,
                result_text_col,
            ],
            api_name="query_image",
        )

        # Caption task handler
        submit_button_caption.click(
            fn=process_caption_submission,
            inputs=[
                model_path_dropdown,
                main_image_uploader,
                caption_length_selector,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
            ],
            outputs=[
                result_image_display,
                result_task_display,
                result_prompt_display,
                result_text_output,
                result_image_col,
                result_text_col,
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
                result_image_col,
                result_text_col,
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
                result_image_col,
                result_text_col,
            ],
        )
    return demo
