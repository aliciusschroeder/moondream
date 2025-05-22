import builtins
import importlib
import numpy as np
import torch
from unittest import mock
from moondream.torch.image_crops import (
    overlap_crop_image,
    reconstruct_from_crops,
    select_tiling,
)


def test_overlap_crop_basic():
    # Create a test image
    test_image = np.zeros((800, 600, 3), dtype=np.uint8)
    # Add a recognizable pattern - white rectangle in the middle
    test_image[300:500, 200:400] = 255

    result = overlap_crop_image(test_image, overlap_margin=4, max_crops=12)

    # Check basic properties
    assert result["crops"][0].shape == (378, 378, 3)
    assert len(result["crops"]) > 1
    assert all(crop.shape == (378, 378, 3) for crop in result["crops"])
    assert len(result["tiling"]) == 2


def test_overlap_crop_small_image():
    # Test with image smaller than crop size
    test_image = np.zeros((300, 200, 3), dtype=np.uint8)
    result = overlap_crop_image(test_image, overlap_margin=4, max_crops=12)

    # Should still produce valid output
    assert result["crops"][0].shape == (378, 378, 3)
    assert len(result["crops"]) == 2
    assert result["tiling"] == (1, 1)


def test_reconstruction():
    # Create a test image
    test_image = np.zeros((800, 600, 3), dtype=np.uint8)
    # Add a recognizable pattern
    test_image[300:500, 200:400] = 255

    # Crop and reconstruct
    result = overlap_crop_image(test_image, overlap_margin=4, max_crops=12)
    crops_tensor = [torch.from_numpy(crop) for crop in result["crops"][1:]]
    reconstructed = reconstruct_from_crops(
        crops_tensor, result["tiling"], overlap_margin=4
    )

    # Convert back to numpy for comparison
    reconstructed_np = reconstructed.numpy()

    # The reconstructed image should be similar to the input
    # We can't expect exact equality due to resizing operations
    # but the white rectangle should still be visible in the middle
    center_reconstructed = reconstructed_np[
        reconstructed_np.shape[0] // 2 - 100 : reconstructed_np.shape[0] // 2 + 100,
        reconstructed_np.shape[1] // 2 - 100 : reconstructed_np.shape[1] // 2 + 100,
    ].mean()

    # The center region should be significantly brighter than the edges
    assert center_reconstructed > reconstructed_np[:100, :100].mean() + 100

def test_select_tiling():
    # Test with both dimensions smaller than crop size
    # both dims <= crop_size → always (1,1)
    assert select_tiling(100, 100, 200, 10) == (1, 1)
    
    # dims > crop_size but min_h*min_w (ceil(500/100)^2=25) > max_crops → fallback ratio branch
    h_tiles, w_tiles = select_tiling(500, 500, 100, 5)
    # ratio = sqrt(5/(5*5)) ~0.447 → floor(5*0.447)=2
    assert (h_tiles, w_tiles) == (2, 2)

    # Perfect aspect-ratio tiles that satisfy max_crops
    # min_h*min_w = 4 and max_crops=4 → perfect‐aspect branch
    assert select_tiling(400, 400, 200, 4) == (2, 2)

    # Exceeding max_crops with w > h
    # w_tiles > h_tiles and h_tiles*w_tiles > max_crops → scales down w_tiles
    assert select_tiling(height=110, width=310, crop_size=100, max_crops=9) == (2, 4)

    # Exceeding max_crops with h > w
    # h_tiles > w_tiles and h_tiles*w_tiles > max_crops → scales down h_tiles
    assert select_tiling(height=310, width=110, crop_size=100, max_crops=9) == (4, 2)

def test_overlap_crop_non_square():
    # non-square image → tiling_h != tiling_w, and crop count = tiling_h*tiling_w + 1
    test_image = np.zeros((800, 400, 3), dtype=np.uint8)
    result = overlap_crop_image(test_image, overlap_margin=4, max_crops=12)
    th, tw = result["tiling"]
    assert th != tw
    assert len(result["crops"]) == th * tw + 1


def test_reconstruct_small_image():
    # small image yields a single local crop → reconstruct to exactly base_size
    test_image = np.zeros((300, 200, 3), dtype=np.uint8)
    result = overlap_crop_image(test_image, overlap_margin=4, max_crops=12)
    # drop the global crop
    local_crops = [torch.from_numpy(c) for c in result["crops"][1:]]
    recon = reconstruct_from_crops(local_crops, result["tiling"], overlap_margin=4)
    assert tuple(recon.shape) == (378, 378, 3)


def test_pil_fallback_branch(monkeypatch):
    original_import = builtins.__import__
    def mocked_import(name, *args, **kwargs):
        if name == "pyvips":
            raise ImportError("Simulated ImportError for pyvips")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=mocked_import):
        # sys.modules.pop("moondream.torch.image_crops", None)
        from moondream.torch import image_crops
        importlib.reload(image_crops)
        monkeypatch.setattr(image_crops, 'HAS_VIPS', False)
        # random image to trigger the PIL branch
        test_image = np.random.randint(0, 256, (800, 800, 3), dtype=np.uint8)
        result = image_crops.overlap_crop_image(test_image, overlap_margin=4, max_crops=4)
        # must still produce a 378×378 global crop via PIL
        assert result['crops'][0].shape == (378, 378, 3)
        assert isinstance(result['tiling'], tuple)
