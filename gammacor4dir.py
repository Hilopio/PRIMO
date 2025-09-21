import numpy as np
from skimage import io
from pathlib import Path
import shutil

# Global variables
input_dir = Path("/home/g.nikolaev/data/MIST-Phase-Contrast-55x55/")
output_dir = Path("/home/g.nikolaev/data/MIST-Phase-Contrast-55x55_gamma/")
gamma = 0.4
percentiles = (0.25, 99.75)


def _make_or_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def process_images():
    # Create or clean output directory
    _make_or_clean_dir(output_dir)

    # Get list of TIFF files with case-insensitive extensions (.tif, .tiff, .TIF, .TIFF, etc.)
    tiff_files = list(input_dir.glob("*.tif"))
    if not tiff_files:
        print("No TIFF files found in input directory.")
        return

    # Print found files for verification
    print("Found files:", [f.name for f in tiff_files])

    # Read all images
    images = [io.imread(file, as_gray=True).astype(np.float32) for file in tiff_files]

    # Compute global percentiles
    all_pixels = np.concatenate([img.ravel() for img in images])
    p_low, p_high = np.percentile(all_pixels, percentiles)

    # Process each image
    for img, file in zip(images, tiff_files):
        # Clip and normalize
        img_clipped = np.clip(img, p_low, p_high)
        normalized = (img_clipped - p_low) / (p_high - p_low)

        # Apply gamma correction
        gamma_corrected = np.power(normalized, gamma)

        # Convert back to 16-bit for saving
        gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

        # Save to output directory
        output_path = output_dir / file.name
        io.imsave(output_path, gamma_corrected)
        print(f"Processed and saved: {output_path}")

    print(f"Global {percentiles[0]}th percentile: {p_low:.2f}")
    print(f"Global {percentiles[1]}th percentile: {p_high:.2f}")


if __name__ == "__main__":
    process_images()
