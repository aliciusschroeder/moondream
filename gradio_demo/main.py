"""
Entry point. Starts the Gradio app. Handles __main__ block and demo.launch(...).
"""

# Import core modules
from .core.config import MODEL_DIR, DEBUG_MODE, SKIP_MODEL_LOAD, get_config_dict
from .core.model_loader import get_model_files_from_directory, initialize_model

# Import utils
from .utils.logging_utils import configure_logging, log_startup_info, log_model_info

# Import UI layout
from .ui.layout import create_gradio_ui


def main():
    """Main entry point for the Gradio application."""
    # Configure logging
    logger = configure_logging(debug_mode=DEBUG_MODE)

    # Log startup information
    log_startup_info(logger, get_config_dict())

    # Initialize model
    model_files_list, initial_model_status, model_loaded = initialize_model(MODEL_DIR, skip_model_load=SKIP_MODEL_LOAD)

    # Log model information
    log_model_info(logger, model_files_list)

    if not model_files_list:
        logger.critical(
            f"CRITICAL WARNING: No model files were found in the specified directory: {MODEL_DIR}."
        )

    # Create Gradio interface
    demo = create_gradio_ui(model_files_list, initial_model_status, model_loaded, logger)

    # Launch the interface
    logger.info("Launching Gradio interface...")
    demo.launch(debug=DEBUG_MODE)

    return demo


# If this script is run directly, launch the app
if __name__ == "__main__":
    main()
