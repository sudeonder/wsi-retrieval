from openslide import OpenSlide
from PIL import Image
import numpy as np
import os
from pathlib import Path

def compute_occupancy_from_pil(img: Image.Image, threshold: int = 210) -> float:
    """Compute occupancy from a PIL image (grayscale)."""
    img_gray = img.convert("L")
    img_np = np.array(img_gray)
    mask = img_np < threshold  # tissue = darker than threshold
    return np.sum(mask) / mask.size

def compute_slide_occupancy(slide: OpenSlide, level: int = 2, threshold: int = 210) -> float:
    """Estimate whole-slide occupancy from low-res thumbnail."""
    thumb = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("L")
    img_np = np.array(thumb)
    mask = img_np < threshold
    return np.sum(mask) / mask.size

def tile_wsi_if_occupied(
    wsi_path: Path,
    output_dir: Path,
    slide_occupancy_threshold=0.1,
    patch_occupancy_threshold=0.5,
    patch_size=1000,
    stride=1000,
    level=0
):
    """Tiles a WSI only if slide-level and patch-level occupancy thresholds are met."""
    slide = OpenSlide(str(wsi_path))
    slide_name = wsi_path.stem

    # Slide-level occupancy check
    occupancy = compute_slide_occupancy(slide)
    print(f"📊 Slide occupancy for {slide_name}: {occupancy:.2%}")
    if occupancy < slide_occupancy_threshold:
        print(f"⚠️ Skipping {slide_name} due to low slide-level occupancy.")
        return

    # Proceed to patching
    width, height = slide.level_dimensions[level]
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    total_patches = 0

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            total_patches += 1
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            occ = compute_occupancy_from_pil(patch)

            if occ >= patch_occupancy_threshold:
                patch.save(output_dir / f"{x}_{y}.png")
                saved_count += 1

    print(f"✅ {saved_count} / {total_patches} patches saved for {slide_name} ({100.0 * saved_count / total_patches:.2f}%)\n")
