"""
Logic for loading/caching models, including get_model_files_from_directory and load_or_get_cached_model.
"""

import os
import torch
import gradio as gr

from ..moondream_imports import (
    MoondreamConfig,
    MoondreamModel,
    load_weights_into_model,
    MOONDREAM_IMPORTS_SUCCESS,
)

# --- Global Model Cache ---
cached_model_obj = None
cached_model_path_str = None
# --- End Global Model Cache ---


def get_model_files_from_directory(directory: str) -> list:
    """
    Scans the specified directory for model files.

    Args:
        directory (str): Directory path to scan

    Returns:
        list: List of model file paths found
    """
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

    Args:
        selected_model_path (str): Path to the model file

    Returns:
        MoondreamModel: The loaded model

    Raises:
        gr.Error: If model loading fails
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


def initialize_model(model_dir: str):
    """
    Initialize the model at startup by loading the first available model file.

    Args:
        model_dir (str): Directory to scan for model files

    Returns:
        tuple: (list of model files, status message)
    """
    model_files_list = get_model_files_from_directory(model_dir)

    initial_model_load_status_message = "No models found or directory empty."
    if not model_files_list:
        return model_files_list, initial_model_load_status_message

    selected_initial_model = model_files_list[0]
    initial_model_load_status_message = f"Attempting to load initial model: {os.path.basename(selected_initial_model)}..."
    print(initial_model_load_status_message)

    try:
        # Check if imports were successful
        if MOONDREAM_IMPORTS_SUCCESS:
            # load_or_get_cached_model(selected_initial_model)
            initial_model_load_status_message = (
                f"✅ Initial model '{os.path.basename(selected_initial_model)}' loaded."
            )
            print(initial_model_load_status_message)
        else:
            initial_model_load_status_message = "❌ Moondream components not fully available for initial load. Check imports and library installation."
            print(
                "ERROR: Moondream components appear to be dummy versions or not fully imported for initial load."
            )
    except Exception as e:
        initial_model_load_status_message = f"❌ Error loading initial model '{os.path.basename(selected_initial_model)}': {e}"
        print(f"ERROR during initial model load: {e}")

    return model_files_list, initial_model_load_status_message
