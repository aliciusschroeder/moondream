"""
Logging utilities for the Gradio application.
"""
import logging
import sys


def configure_logging(debug_mode=False):
    """
    Configure logging for the application.
    
    Args:
        debug_mode (bool): Whether to enable debug logging
    """
    # Set up logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging level based on debug mode
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create a logger specific to this application
    logger = logging.getLogger('moondream_gradio')
    logger.setLevel(log_level)
    
    # Return the logger
    return logger


def log_startup_info(logger, config_dict):
    """
    Log application startup information.
    
    Args:
        logger (logging.Logger): Logger to use
        config_dict (dict): Configuration dictionary
    """
    logger.info("Starting Moondream Gradio Interface...")
    
    # Log configuration
    logger.info("Configuration:")
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")
        
    # Log import status
    from ..moondream_imports import MOONDREAM_IMPORTS_SUCCESS
    if MOONDREAM_IMPORTS_SUCCESS:
        logger.info("Moondream imports successful")
    else:
        logger.warning("Moondream imports failed - using fallback dummy classes")


def log_model_info(logger, model_files_list):
    """
    Log information about available models.
    
    Args:
        logger (logging.Logger): Logger to use
        model_files_list (list): List of model file paths
    """
    if not model_files_list:
        logger.critical("No model files found. The application may not function correctly.")
    else:
        logger.info(f"Found {len(model_files_list)} model files:")
        for model_file in model_files_list:
            logger.info(f"  {model_file}")


def log_error(logger, message, exception=None):
    """
    Log an error message and optionally the exception.
    
    Args:
        logger (logging.Logger): Logger to use
        message (str): Error message
        exception (Exception, optional): Exception to log
    """
    if exception:
        logger.error(f"{message}: {str(exception)}", exc_info=True)
    else:
        logger.error(message)

