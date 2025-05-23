"""
Defines the full Gradio UI layout and returns the interface demo.
"""

import gradio as gr

from .events_suggestions import (
    object_suggestions_event,
    question_suggestions_event,
    handle_suggestion_click,
)
from .events_tasks import (
    process_caption_submission,
    process_point_submission,
    process_query_submission,
    process_detect_submission,
    process_detect_all_submission,
)
from .events import (
    handle_detectall_max_tiles_change,
    handle_model_selection_change,
)
from ..utils.ui_utils import create_model_choices
from ..core.config import APP_TITLE, APP_DESCRIPTION, PRIMARY_HUE, SECONDARY_HUE
from ..moondream_imports import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_OBJECTS,
)


def create_gradio_ui(model_files_list, initial_model_status, model_loaded, logger):
    """
    Create and configure the Gradio UI.

    Args:
        model_files_list (list): List of model file paths
        initial_model_status (str): Initial model loading status message

    Returns:
        gr.Blocks: The configured Gradio interface
    """
    logger.info("Creating Gradio UI...")
    # Create the Gradio Blocks interface
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue=PRIMARY_HUE, secondary_hue=SECONDARY_HUE)
    ) as demo:
        gr.Markdown(APP_DESCRIPTION)

        # Hidden state variables for tracking
        last_processed_image = gr.State(
            {
                "query_tab": None,
                "point_tab": None,
                "detect_tab": None,
            }
        )  # Store hash of last processed image
        current_tab = gr.State("query_tab")  # Track current tab
        point_tab_id = gr.State("point_tab")
        detect_tab_id = gr.State("detect_tab")

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
                        with gr.Row() as query_suggestion_status_row:
                            query_suggestion_status = gr.Markdown(
                                "*Upload an image to get question suggestions*",
                                visible=True,
                            )

                        with gr.Row() as query_button_row:
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
                        with gr.Row() as point_suggestion_status_row:
                            point_suggestion_status = gr.Markdown(
                                "*Upload an image to get point suggestions*",
                                visible=True,
                            )
                        with gr.Row() as points_buttons_row:
                            point_suggestion_btn1 = gr.Button(
                                "Suggestion 1", variant="secondary", visible=False
                            )
                            point_suggestion_btn2 = gr.Button(
                                "Suggestion 2", variant="secondary", visible=False
                            )
                            point_suggestion_btn3 = gr.Button(
                                "Suggestion 3", variant="secondary", visible=False
                            )
                        submit_button_point = gr.Button("SUBMIT", variant="primary")

                    with gr.TabItem("👁️ Detect", id="detect_tab") as detect_tab:
                        gr.Markdown(
                            "Detect objects. Settings from 'Object Settings' will apply."
                        )
                        object_textbox_detect = gr.Textbox(
                            label="Object to Detect", placeholder="e.g., cat, table"
                        )
                        with gr.Row() as detect_suggestion_status_row:
                            detect_suggestion_status = gr.Markdown(
                                "*Upload an image to get detect suggestions*",
                                visible=True,
                            )
                        with gr.Row() as detect_buttons_row:
                            detect_suggestion_btn1 = gr.Button(
                                "Suggestion 1", variant="secondary", visible=False
                            )
                            detect_suggestion_btn2 = gr.Button(
                                "Suggestion 2", variant="secondary", visible=False
                            )
                            detect_suggestion_btn3 = gr.Button(
                                "Suggestion 3", variant="secondary", visible=False
                            )
                        submit_button_detect = gr.Button("SUBMIT", variant="primary")

                    with gr.TabItem(
                        "🔍 Detect All", id="detect_all_tab"
                    ) as detect_all_tab:
                        gr.Markdown(
                            "Detect all objects in the image. This may take a while."
                        )
                        in_depth_checkbox = gr.Checkbox(
                            label="In-depth analysis",
                            value=True,
                            info="Enable in-depth analysis for better detection.",
                        )
                        max_tiles_slider = gr.Slider(
                            minimum=2,
                            maximum=100,
                            value=12,
                            step=1,
                            label="Max Tiles",
                            info="Maximum number of tiles for detection.",
                            visible=True,
                        )
                        actual_tiles_indicator = gr.Markdown(
                            "Upload an image to see the number of tiles.",
                        )
                        submit_button_detect_all = gr.Button(
                            "SUBMIT", variant="primary"
                        )

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
                model_load_button = gr.Button(
                    "Load Model",
                    variant="primary",
                    visible=False if model_loaded else True,
                )

            with gr.Accordion("Text Generation", open=True):  # Collapsible sub-section
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

            with gr.Accordion("Object Detection", open=True):  # Collapsible sub-section
                max_objects_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=DEFAULT_MAX_OBJECTS,
                    step=1,
                    label="Max Objects",
                    info="Max objects for detection/pointing tasks.",
                )

        # --- Event Handlers ---
        logger.info("Setting up event handlers...")
        # Model Selection
        model_path_dropdown.change(
            fn=handle_model_selection_change,
            inputs=[model_path_dropdown],
            outputs=[model_load_status_md, model_load_button],
        )
        model_load_button.click(
            fn=handle_model_selection_change,
            inputs=[model_path_dropdown],
            outputs=[model_load_status_md, model_load_button],
        )

        # Update tab state
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
        detect_all_tab.select(
            lambda: "detect_all_tab",
            inputs=None,
            outputs=[current_tab],
        )

        # Suggestion Trigger on Tab Change
        current_tab.change(
            fn=question_suggestions_event,
            inputs=[
                model_path_dropdown,
                main_image_uploader,
                last_processed_image,
                current_tab,
            ],
            outputs=[
                query_suggestion_status,
                query_suggestion_btn1,
                query_suggestion_btn2,
                query_suggestion_btn3,
                last_processed_image,
            ],
        ).then(
            fn=object_suggestions_event,
            inputs=[
                model_path_dropdown,
                main_image_uploader,
                last_processed_image,
                current_tab,
                point_tab_id,
            ],
            outputs=[
                point_suggestion_status,
                point_suggestion_btn1,
                point_suggestion_btn2,
                point_suggestion_btn3,
                last_processed_image,
            ],
        ).then(
            fn=object_suggestions_event,
            inputs=[
                model_path_dropdown,
                main_image_uploader,
                last_processed_image,
                current_tab,
                detect_tab_id,
            ],
            outputs=[
                detect_suggestion_status,
                detect_suggestion_btn1,
                detect_suggestion_btn2,
                detect_suggestion_btn3,
                last_processed_image,
            ],
        )

        # Suggestion Trigger on Image Upload
        main_image_uploader.change(
            fn=question_suggestions_event,
            inputs=[
                model_path_dropdown,
                main_image_uploader,
                last_processed_image,
                current_tab,
            ],
            outputs=[
                query_suggestion_status,
                query_suggestion_btn1,
                query_suggestion_btn2,
                query_suggestion_btn3,
                last_processed_image,
            ],
        ).then(
            fn=object_suggestions_event,
            inputs=[
                model_path_dropdown,
                main_image_uploader,
                last_processed_image,
                current_tab,
                point_tab_id,
            ],
            outputs=[
                point_suggestion_status,
                point_suggestion_btn1,
                point_suggestion_btn2,
                point_suggestion_btn3,
                last_processed_image,
            ],
        ).then(
            fn=object_suggestions_event,
            inputs=[
                model_path_dropdown,
                main_image_uploader,
                last_processed_image,
                current_tab,
                detect_tab_id,
            ],
            outputs=[
                detect_suggestion_status,
                detect_suggestion_btn1,
                detect_suggestion_btn2,
                detect_suggestion_btn3,
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

        # Make suggestion buttons update the point input when clicked
        point_suggestion_btn1.click(
            fn=handle_suggestion_click,
            inputs=[point_suggestion_btn1],
            outputs=[object_textbox_point],
        )
        point_suggestion_btn2.click(
            fn=handle_suggestion_click,
            inputs=[point_suggestion_btn2],
            outputs=[object_textbox_point],
        )
        point_suggestion_btn3.click(
            fn=handle_suggestion_click,
            inputs=[point_suggestion_btn3],
            outputs=[object_textbox_point],
        )

        # Make suggestion buttons update the detect input when clicked
        detect_suggestion_btn1.click(
            fn=handle_suggestion_click,
            inputs=[detect_suggestion_btn1],
            outputs=[object_textbox_detect],
        )
        detect_suggestion_btn2.click(
            fn=handle_suggestion_click,
            inputs=[detect_suggestion_btn2],
            outputs=[object_textbox_detect],
        )
        detect_suggestion_btn3.click(
            fn=handle_suggestion_click,
            inputs=[detect_suggestion_btn3],
            outputs=[object_textbox_detect],
        )

        # Elements of "Detect All"
        in_depth_checkbox.change(
            fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=[in_depth_checkbox],
            outputs=[max_tiles_slider, actual_tiles_indicator],
        )

        # Update max tiles indicator
        max_tiles_slider.change(
            fn=handle_detectall_max_tiles_change,
            inputs=[main_image_uploader, in_depth_checkbox, max_tiles_slider],
            outputs=[actual_tiles_indicator],
        )

        # Submission Handlers
        # Query task handler
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
            fn=process_point_submission,
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
            fn=process_detect_submission,
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

        # Detect All task handler
        submit_button_detect_all.click(
            fn=process_detect_all_submission,
            inputs=[
                model_path_dropdown,
                main_image_uploader,
                max_objects_slider,
                in_depth_checkbox,
                max_tiles_slider,
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
