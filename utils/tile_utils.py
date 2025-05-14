from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
import openslide
import json

def calculate_occupancy(patch: np.ndarray, use_otsu=True, threshold=0.8):
    """
    Calculate the tissue occupancy of a patch.
    If use_otsu is True, applies Otsu thresholding. Otherwise uses a fixed threshold.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    if use_otsu:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)

    foreground_fraction = 1.0 - (np.sum(binary == 255) / binary.size)
    return foreground_fraction


def tile_wsi_if_occupied(
    wsi_path: Path,
    output_dir: Path,
    patch_occupancy_threshold: float = 0.5,
    patch_size: int = 448,
    stride: int = 448,
    resize_dim: int = 224,
    level: int = 0,
    use_otsu: bool = True,
    generate_thumbnail: bool = True,
    debug: bool = False,
):
    """
    Tiles a WSI and saves patches above a tissue occupancy threshold.
    Returns a list of metadata dicts for each saved patch.
    Also generates a thumbnail visualizing saved patch locations.
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

    # Create thumbnail for debug/visualization
    if generate_thumbnail:
        thumb = slide.get_thumbnail((width // 32, height // 32)).convert("RGB")
        draw = ImageDraw.Draw(thumb)
        scale_x = thumb.width / width
        scale_y = thumb.height / height

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(patch)

            occ = calculate_occupancy(patch_np, use_otsu, threshold=0.8)

            if occ < patch_occupancy_threshold:
                total_patches += 1
                continue

            # Resize and save
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

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(dataset_records, f, indent=2)

    # Save thumbnail with overlay
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
