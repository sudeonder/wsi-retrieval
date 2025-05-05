from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import openslide

def calculate_occupancy(patch, threshold=0.8):
    """Calculate the occupancy of a patch (non-white pixels)."""
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
    occupancy = 1.0 - (np.sum(binary == 255) / binary.size)
    return occupancy

def tile_wsi_if_occupied(
    wsi_path: Path,
    output_dir: Path,
    slide_occupancy_threshold=0.1,
    patch_occupancy_threshold=0.5,
    patch_size=448,  # Extract at 40x
    stride=448,
    resize_dim=224,  # Downsample to simulate 20x
    level=0
):
    slide = openslide.OpenSlide(str(wsi_path))
    width, height = slide.level_dimensions[level]

    basename = wsi_path.stem
    output_dir = output_dir / basename
    output_dir.mkdir(parents=True, exist_ok=True)

    total_patches = 0
    saved_patches = 0

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            patch = np.array(patch)

            slide_occ = calculate_occupancy(patch, threshold=0.8)
            if slide_occ < patch_occupancy_threshold:
                continue

            # Resize to simulate 20x
            patch_resized = cv2.resize(patch, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)
            patch_img = Image.fromarray(patch_resized)

            tile_name = f"{x}_{y}.png"
            patch_img.save(output_dir / tile_name)
            saved_patches += 1

            total_patches += 1

    print(f"✅ Done: {saved_patches} / {total_patches} patches saved for {basename}")
    slide.close()
