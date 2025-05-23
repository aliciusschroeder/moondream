"""
Vision encoder module for processing images using Vision Transformer architecture.

This module provides functionality for encoding images into feature representations
using patch-based processing, attention mechanisms, and projection layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union, Tuple
from PIL import Image

from .layers import attn, layer_norm, mlp
from .image_crops import overlap_crop_image
from .config import VisionConfig

if torch.backends.mps.is_available():
    # Non-divisible input sizes are not implemented on MPS device yet.
    # https://github.com/pytorch/pytorch/issues/96056
    def adaptive_avg_pool2d(input, output_size):
        return F.adaptive_avg_pool2d(input.to("cpu"), output_size).to("mps")

else:
    adaptive_avg_pool2d = F.adaptive_avg_pool2d

DeviceLike = Union[str, torch.device, int]


def prepare_crops(
    image: Image.Image, config: VisionConfig, device: DeviceLike
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Prepare image crops for vision processing.
    
    Args:
        image: PIL Image to process
        config: Vision configuration containing crop settings
        device: Target device for tensor operations
        
    Returns:
        Tuple containing:
            - Processed tensor of image crops with shape [N, C, H, W]
            - Tiling information as (rows, cols)
    """
    np_image = np.array(image.convert("RGB"))
    overlap_crops = overlap_crop_image(
        np_image, max_crops=config.max_crops, overlap_margin=config.overlap_margin
    )
    all_crops = overlap_crops["crops"]
    all_crops = np.transpose(all_crops, (0, 3, 1, 2))
    all_crops = (
        torch.from_numpy(all_crops)
        .to(device=device, dtype=torch.bfloat16)
        .div_(255.0)
        .sub_(0.5)
        .div_(0.5)
    )
    return all_crops, overlap_crops["tiling"]


def create_patches(x, patch_size):
    """
    Convert image tensor into patches for Vision Transformer processing.
    
    Takes an input image tensor and reshapes it into non-overlapping patches
    that can be processed by the transformer architecture.
    
    Args:
        x: Input tensor with shape [B, C, H, W]
        patch_size: Size of each square patch
        
    Returns:
        Tensor with shape [B, (H/patch_size)*(W/patch_size), C*patch_size*patch_size]
        where each patch is flattened into a vector
    """
    # Original shape: [B, C, H, W]
    B, C, H, W = x.shape
    P1 = P2 = patch_size

    # Step 1: Split H and W dimensions into patches
    # [B, C, H/P1, P1, W/P2, P2]
    x = x.reshape(B, C, H // P1, P1, W // P2, P2)

    # Step 2: Rearrange dimensions to match target shape
    # [B, H/P1, W/P2, C, P1, P2]
    x = x.permute(0, 2, 4, 1, 3, 5)

    # Step 3: Combine dimensions to get final shape
    # [B, (H/P1)*(W/P2), C*P1*P2]
    x = x.reshape(B, (H // P1) * (W // P2), C * P1 * P2)

    return x


def vision_encoder(input_BCHW: torch.Tensor, w: nn.Module, config: VisionConfig):
    """
    Encode image patches using Vision Transformer architecture.
    
    Processes image patches through patch embedding, positional encoding,
    and multiple transformer blocks with attention and MLP layers.
    
    Args:
        input_BCHW: Input image tensor with shape [B, C, H, W]
        w: Neural network weights module containing transformer components
        config: Vision configuration with encoder parameters
        
    Returns:
        Encoded feature tensor after processing through all transformer blocks
    """
    x = create_patches(input_BCHW, config.enc_patch_size)

    x = w.patch_emb(x)
    x = x + w.pos_emb
    for block in w.blocks:
        x = x + attn(layer_norm(x, block.ln1), block.attn, n_heads=config.enc_n_heads)
        x = x + mlp(layer_norm(x, block.ln2), block.mlp)
    x = layer_norm(x, w.post_ln)

    return x


def vision_projection(
    global_features: torch.Tensor,
    reconstructed: torch.Tensor,
    w: nn.Module,
    config: VisionConfig,
):
    """
    Project vision features to final output dimension.
    
    Combines global features with reconstructed features and projects them
    through an MLP to produce the final vision representation.
    
    Args:
        global_features: Global image features tensor
        reconstructed: Reconstructed feature tensor with spatial information
        w: Neural network weights module containing projection MLP
        config: Vision configuration with projection parameters
        
    Returns:
        Final projected features ready for downstream tasks
    """
    reconstructed = reconstructed.permute(2, 0, 1)
    reconstructed = adaptive_avg_pool2d(
        reconstructed, output_size=(config.enc_n_layers, config.enc_n_layers)
    )
    reconstructed = reconstructed.permute(1, 2, 0).view(729, config.enc_dim)
    final_features = torch.cat([global_features, reconstructed], dim=-1)
    return mlp(final_features, w.proj_mlp)


def build_vision_model(config: VisionConfig, dtype: torch.dtype):
    """
    Build the complete vision model architecture.
    
    Constructs all components of the vision transformer including patch embedding,
    transformer blocks with attention and MLP layers, layer normalization,
    and projection layers.
    
    Args:
        config: Vision configuration containing all model parameters
        dtype: PyTorch data type for model parameters
        
    Returns:
        nn.ModuleDict containing all vision model components:
            - patch_emb: Linear layer for patch embedding
            - blocks: List of transformer blocks
            - post_ln: Final layer normalization
            - proj_mlp: Projection MLP
            - pos_emb: Positional embedding parameters
    """
    patch_dim = config.enc_patch_size * config.enc_patch_size * config.in_channels
    grid_size = config.crop_size // config.enc_patch_size
    num_patches = grid_size * grid_size

    vision = nn.ModuleDict(
        {
            "patch_emb": nn.Linear(patch_dim, config.enc_dim, dtype=dtype),
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln1": nn.LayerNorm(config.enc_dim, dtype=dtype),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": nn.Linear(
                                        config.enc_dim, 3 * config.enc_dim, dtype=dtype
                                    ),
                                    "proj": nn.Linear(
                                        config.enc_dim, config.enc_dim, dtype=dtype
                                    ),
                                }
                            ),
                            "ln2": nn.LayerNorm(config.enc_dim, dtype=dtype),
                            "mlp": nn.ModuleDict(
                                {
                                    "fc1": nn.Linear(
                                        config.enc_dim, config.enc_ff_dim, dtype=dtype
                                    ),
                                    "fc2": nn.Linear(
                                        config.enc_ff_dim, config.enc_dim, dtype=dtype
                                    ),
                                }
                            ),
                        }
                    )
                    for _ in range(config.enc_n_layers)
                ]
            ),
            "post_ln": nn.LayerNorm(config.enc_dim, dtype=dtype),
            "proj_mlp": nn.ModuleDict(
                {
                    "fc1": nn.Linear(
                        config.enc_dim * 2, config.proj_inner_dim, dtype=dtype
                    ),
                    "fc2": nn.Linear(
                        config.proj_inner_dim, config.proj_out_dim, dtype=dtype
                    ),
                }
            ),
        }
    )
    vision.pos_emb = nn.Parameter(
        torch.zeros(1, num_patches, config.enc_dim, dtype=dtype)
    )
    return vision
