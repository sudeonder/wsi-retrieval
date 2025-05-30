from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
import openslide
import json

def calculate_occupancy(patch: np.ndarray, min_saturation: int = 15):
    """
    Estimate tissue occupancy using the HSV saturation channel.
    Keeps pale pink tissue with low brightness but detectable color.

    Parameters:
    - patch: RGB image as numpy array
    - min_saturation: pixels with S > this value are considered tissue

    Returns:
    - occupancy: fraction of tissue-colored pixels
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]  # S channel
    tissue_mask = saturation > min_saturation
    occupancy = np.sum(tissue_mask) / tissue_mask.size
    return occupancy


def tile_wsi_if_occupied(
    wsi_path: Path,
    output_dir: Path,
    patch_occupancy_threshold: float = 0.5,
    patch_size: int = 448,
    stride: int = 448,
    resize_dim: int = 224,
    level: int = 0,
    min_saturation: int = 15,
    generate_thumbnail: bool = True,
    debug: bool = False,
):
    """
    Tiles a WSI and saves patches with sufficient saturation-based tissue occupancy.
    Saves metadata and thumbnail for each WSI.
    """
    slide = openslide.OpenSlide(str(wsi_path))
    width, height = slide.level_dimensions[level]
    basename = wsi_path.stem
    output_dir = output_dir / basename
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_records = []
    saved_coords = []
    total_patches = 0
    saved_patches = 0

    if generate_thumbnail:
        thumb = slide.get_thumbnail((width // 32, height // 32)).convert("RGB")
        draw = ImageDraw.Draw(thumb)
        scale_x = thumb.width / width
        scale_y = thumb.height / height

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(patch)

            occ = calculate_occupancy(patch_np, min_saturation=min_saturation)

            if occ < patch_occupancy_threshold:
                total_patches += 1
                continue

            patch_resized = cv2.resize(patch_np, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)
            patch_img = Image.fromarray(patch_resized)
            tile_name = f"x{x}_y{y}.png"
            patch_img.save(output_dir / tile_name)

            dataset_records.append({
                "x": x,
                "y": y,
                "tile_filename": tile_name,
                "occupancy": occ,
                "saved": True
            })

            saved_coords.append((x, y))
            saved_patches += 1
            total_patches += 1

            if debug:
                print(f"[{tile_name}] occupancy={occ:.3f} → saved.")

    slide.close()

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(dataset_records, f, indent=2)

    if generate_thumbnail and saved_coords:
        box_size = int(patch_size * scale_x)
        for x, y in saved_coords:
            draw.rectangle(
                [
                    (int(x * scale_x), int(y * scale_y)),
                    (int((x + patch_size) * scale_x), int((y + patch_size) * scale_y))
                ],
                outline="red",
                width=1
            )
        thumb.save(output_dir / "thumbnail.png")

    print(f"✅ Done: {saved_patches} / {total_patches} patches saved for {basename}")
    return dataset_records
