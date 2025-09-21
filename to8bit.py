from pathlib import Path
import cv2
import numpy as np
from src.utils import _make_or_clean_dir

# Глобальные переменные
INPUT_DIR = Path("/home/g.nikolaev/data/MIST/Phase_Small")
OUTPUT_DIR = Path("/home/g.nikolaev/data/normalized_MIST/Phase_Small")
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
USE_CLAHE = True


def convert_16bit_to_8bit():
    if not INPUT_DIR.exists():
        raise ValueError(f"Input directory {INPUT_DIR} does not exist")

    _make_or_clean_dir(OUTPUT_DIR)

    # Собираем TIFF-файлы
    image_paths = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in (".tif", ".tiff")]
    if not image_paths:
        raise ValueError(f"No TIFF images found in {INPUT_DIR}")

    # Глобальные перцентили
    intensities = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Could not load {path}")
            continue
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        intensities.append(img.flatten())

    if not intensities:
        raise ValueError("No valid images loaded")

    intensities = np.concatenate(intensities)
    p_min, p_max = np.percentile(intensities, (LOWER_PERCENTILE, UPPER_PERCENTILE))
    print(f"Percentiles ({LOWER_PERCENTILE}%-{UPPER_PERCENTILE}%): {p_min}-{p_max}")

    # CLAHE для больших изображений (1k×1k+)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16)) if USE_CLAHE else None

    # Обработка изображений
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Could not load {path}")
            continue
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint16:
            print(f"Warning: {path} is not 16-bit (dtype: {img.dtype})")
            continue

        img_clipped = np.clip(img, p_min, p_max)
        img_8bit = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if USE_CLAHE:
            img_8bit = clahe.apply(img_8bit)

        cv2.imwrite(str(OUTPUT_DIR / f"{path.stem}.png"), img_8bit)
        print(f"Saved: {path.stem}.png")


if __name__ == "__main__":
    convert_16bit_to_8bit()
