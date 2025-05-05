from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import openslide

def calculate_occupancy(patch, threshold=0.8):
    """
    Calculate the occupancy of a patch based on non-white pixel content.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
    occupancy = 1.0 - (np.sum(binary == 255) / binary.size)
    return occupancy

def tile_wsi_if_occupied(
    wsi_path: Path,
    output_dir: Path,
    patch_occupancy_threshold=0.5,
    patch_size=448,
    stride=448,
    resize_dim=224,
    level=0
):
    """
    Tiles a WSI and saves patches above the occupancy threshold.
    Returns metadata for each saved tile for use in dataset.csv.
    """
    slide = openslide.OpenSlide(str(wsi_path))
    width, height = slide.level_dimensions[level]

    basename = wsi_path.stem
    output_dir = output_dir / basename
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_records = []
    total_patches = 0
    saved_patches = 0

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(patch)

            occ = calculate_occupancy(patch_np, threshold=0.8)
            if occ < patch_occupancy_threshold:
                continue  # Skip low-content (white) patches

            # Resize and save
            patch_resized = cv2.resize(patch_np, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)
            patch_img = Image.fromarray(patch_resized)

            tile_name = f"x{x}_y{y}.png"
            patch_img.save(output_dir / tile_name)

            dataset_records.append({
                "x": x,
                "y": y,
                "tile_filename": tile_name
            })

            saved_patches += 1
            total_patches += 1

    slide.close()
    print(f"✅ Done: {saved_patches} / {total_patches} patches saved for {basename}")

    return dataset_records
