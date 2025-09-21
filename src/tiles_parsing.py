from pathlib import Path
from src.classes import Tile, TileSet
from src.logger import logger
import re

img_extensions = (".jpg", ".png", ".bmp", ".tiff", ".tif", ".TIF", "TIFF")


def get_row_col(img_path):
    stem = img_path.stem
    # match = re.search(r'(\d+)x(\d+)', stem)
    match = re.search(r'(\d+)\.(\d+)', stem)

    if not match:
        match = re.search(r'r(\d+)_c(\d+)', stem)

    if not match:
        raise ValueError(f"Неверный формат имени файла: {img_path.name}. ")

    row = int(match.group(1))
    col = int(match.group(2))
    return row, col


def _tiles_parsing(dir_path: Path, use_grid_info: bool = False) -> TileSet:
    """
    Parse a directory to create an ImageSet object for image files.
    Args:
        dir_path (Path): Path to the directory containing image files.
    Returns:
        ImageSet: An ImageSet object containing the order and dictionary of
            ImageStruct objects representing the images in the directory.
    """
    try:
        img_paths = [
            img_p
            for img_p in dir_path.iterdir()
            if img_p.suffix in img_extensions
        ]
        img_paths.sort(key=lambda x: x.name)

        order, rowcol, images = [], {}, {}
        for id, path in enumerate(img_paths):
            order.append(id)
            if use_grid_info:
                row, col = get_row_col(path)
            else:
                row, col = None, None
            rowcol[id] = (row, col)
            images[id] = Tile(
                id=id,
                img_path=path,
                _image=None,
                _tensor=None,
                orig_size=None,
                homography=None,
                gain=None
            )
    except Exception as e:
        logger.error(f"Error parsing directory: {e}")
        return None

    return TileSet(order=order, rowcol=rowcol, images=images)
