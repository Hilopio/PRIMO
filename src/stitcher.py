from pathlib import Path
import shutil
import numpy as np
from src.logger import logger, log_time

from src.classes import StitchingData, Panorama, AlignConfig

from src.matcher import Matcher
from src.alignment import _align, _CUBA
from src.collage_functions import make_collage, make_mosaic
from src.tiles_parsing import _tiles_parsing

from src.gain_comp_functions import apply_gain_comp
from src.graphcut_functions import apply_graphcut
from src.blending_functions import apply_blending

from src.serializer import Serializer

from src.utils import undistort_dir


class Stitcher:
    """
    A class to handle the stitching of multiple images into a single panorama.
    This class orchestrates the process of image matching, alignment, and composition
    to create a seamless panorama from a set of input images using various techniques
    like homography estimation, bundle adjustment, gain compensation, graphcut, and
    blending.
    """
    def __init__(
        self, matcher: Matcher, load_matches: bool = False, align_cfg: AlignConfig = AlignConfig(),
        save_mean_color: bool = True,
        coarse_scale: int = 4, fine_scale: int = 16, lane_width: int = 200, n_levels: int = 7,
        use_gain_comp: bool = True, use_graphcut: bool = True, use_blending: bool = True,
        detailed_log: bool = True, draw_inliers: bool = False, draw_connections: bool = False,
        use_grid_info: bool = False,
        custom_undistortion: bool = False, n_undistortions: int = 1, stitching_mode: str = "collage"
    ) -> None:
        """
        Initialize the Stitcher with a matcher object and configuration parameters.
        Args:
            matcher (Matcher): An instance of Matcher class used for matching images.
            confidence_tr (float): Confidence threshold for matching. Defaults to 0.95.
            min_inliers (int): Minimum number of inliers required for a match. Defaults to 5.
            max_inliers (int): Maximum number of inliers to consider for a match. Defaults to 30.
            min_inlier_rate (float): Minimum inlier rate for matching. Defaults to 0.0.
            reproj_tr (float): Reprojection threshold for alignment. Defaults to 1.0.
            coarse_scale (int): Coarse scale factor for processing. Defaults to 4.
            fine_scale (int): Fine scale factor for processing. Defaults to 16.
            lane_width (int): Width of the lane for stitching. Defaults to 200.
            n_levels (int): Number of levels for multi-scale processing. Defaults to 7.
        """
        self.device = matcher.device
        self.matcher = matcher

        self.load_matches = load_matches

        self.align_cfg = align_cfg

        self.use_gain_comp = use_gain_comp
        self.use_graphcut = use_graphcut
        self.use_blending = use_blending
        self.save_mean_color = save_mean_color
        self.coarse_scale = coarse_scale
        self.fine_scale = fine_scale
        self.lane_width = lane_width
        self.n_levels = n_levels

        self.detailed_log = detailed_log
        self.draw_inliers = draw_inliers
        self.draw_connections = draw_connections

        self.use_grid_info = use_grid_info
        self.custom_undistortion = custom_undistortion
        self.n_undistortions = n_undistortions
        self.stitching_mode = stitching_mode

    def _compose(self, data: StitchingData, use_gain_comp: bool = True, use_graphcut: bool = True,
                 coarse_scale: int = None, fine_scale: int = None, lane_width: int = None,
                 use_blending: bool = None,  n_levels: int = None, detailed_log: bool = None) -> Panorama:
        """
        Compose a panorama from aligned image data using gain compensation,
        graphcut, and blending techniques.
        Args:
            data (StitchingData): StitchingData object containing aligned image
                information, including transformations and matches.
        Returns:
            PanoramaData: Data object representing the composed panorama with the
                final image and canvas.
        Raises:
            RuntimeError: If the composition process fails due to issues in gain
                compensation, graphcut, or blending.
        """
        use_gain_comp = use_gain_comp if use_gain_comp is not None else self.use_gain_comp
        use_graphcut = use_graphcut if use_graphcut is not None else self.use_graphcut
        use_blending = use_blending if use_blending is not None else self.use_blending

        coarse_scale = coarse_scale if coarse_scale is not None else self.coarse_scale
        fine_scale = fine_scale if fine_scale is not None else self.fine_scale
        lane_width = lane_width if lane_width is not None else self.lane_width
        n_levels = n_levels if n_levels is not None else self.n_levels
        detailed_log = detailed_log if detailed_log is not None else self.detailed_log

        if use_blending and not use_graphcut:
            raise NotImplementedError

        try:
            if use_gain_comp:
                data = apply_gain_comp(data, save_mean_color=True)

            if use_graphcut:
                data = apply_graphcut(data, use_gains=use_gain_comp, coarse_scale=coarse_scale,
                                      fine_scale=fine_scale, lane_width=lane_width)

            if use_blending:
                panorama_data = apply_blending(data, n_levels=n_levels, use_gains=use_gain_comp)

            return panorama_data

        except Exception as e:
            logger.error(f"Composition failed: {str(e)}")
            return None

    def _stitch_full_pipline(self, data: StitchingData) -> Panorama:
        """
        Perform the full stitching pipeline to create a seamless panorama from
        input images.
        Args:
            images (ImageSet): Set of images to be stitched, containing image
                data and processing order.
        Returns:
            PanoramaData: Data object representing the final stitched panorama
                with the composed image and canvas.
        Raises:
            RuntimeError: If any step in the stitching pipeline (alignment or
                composition) fails.
        """
        try:
            alignment_data = _align(
                data=data,
                cfg=self.align_cfg
            )
            panorama_data = self._compose(alignment_data)
            return panorama_data
        except Exception as e:

            logger.error(f"Full stitching pipeline failed: {str(e)}")
            return None

    def _stitch_collage(self, data: StitchingData) -> Panorama:
        """
        Create a collage-style panorama from input images with minimal blending.
        Args:
            images (ImageSet): Set of images to be stitched into a collage,
                containing image data and processing order.
        Returns:
            PanoramaData: Data object representing the stitched collage panorama
                with the composed image and canvas.
        Raises:
            RuntimeError: If any step in the collage stitching process
                (alignment or collage creation) fails.
        """
        try:
            alignment_data = _align(
                data=data,
                cfg=self.align_cfg
            )
            panorama_data = make_collage(
                alignment_data,
                use_gains=False,
                draw_inliers=self.draw_inliers,
                draw_connections=self.draw_connections
            )
            return panorama_data
        except Exception as e:

            logger.error(f"Collage stitching failed: {str(e)}")
            return None

    def _stitch_compensated_collage(self, data: StitchingData) -> Panorama:
        """
        Perform the full stitching pipeline to create a seamless panorama from
        input images.
        Args:
            images (ImageSet): Set of images to be stitched, containing image
                data and processing order.
        Returns:
            PanoramaData: Data object representing the final stitched panorama
                with the composed image and canvas.
        Raises:
            RuntimeError: If any step in the stitching pipeline (alignment or
                composition) fails.
        """
        try:
            alignment_data = _align(data)
            data = apply_gain_comp(alignment_data)
            panorama_data = make_collage(data, use_gains=True)
            return panorama_data
        except Exception as e:

            logger.error(f"Full stitching pipeline failed: {str(e)}")
            return None

    def _stitch_compensated_mosaic(self, data: StitchingData) -> Panorama:
        """
        Perform the full stitching pipeline to create a seamless panorama from
        input images.
        Args:
            images (ImageSet): Set of images to be stitched, containing image
                data and processing order.
        Returns:
            PanoramaData: Data object representing the final stitched panorama
                with the composed image and canvas.
        Raises:
            RuntimeError: If any step in the stitching pipeline (alignment or
                composition) fails.
        """
        try:
            alignment_data = _align(data)
            data = apply_gain_comp(alignment_data)
            data = apply_graphcut(data)
            panorama_data = make_mosaic(data, use_gains=True)
            return panorama_data
        except Exception as e:

            logger.error(f"Full stitching pipeline failed: {str(e)}")
            return None

    def preproccess_undictortion(
        self,
        tiles_dir: Path,
        cache_dir: Path,
        n_undistortions: int = None
    ) -> tuple[np.ndarray, np.ndarray]:

        current_dir = Path(str(tiles_dir))
        tmp_dir = cache_dir / 'tmp'
        for _ in range(n_undistortions):
            tile_set = _tiles_parsing(current_dir, use_grid_info=self.use_grid_info)
            data = self.matcher.match(tile_set, use_grid_info=self.use_grid_info)
            camera_matrix, distortion_params = _CUBA(data, self.align_cfg)

            if iter == n_undistortions - 1:
                break

            undistort_dir(current_dir, tmp_dir, camera_matrix, distortion_params)
            current_dir = tmp_dir

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        return camera_matrix, distortion_params

    @log_time("Panorama done for", logger)
    def stitch(self, tiles_dir: Path, output_file: Path, cache_path: Path = None,
               load_matches: bool = None, mode: str = None) -> None:
        """
        Stitch images from a directory into a panorama with the specified mode
        and save the result to a file.
        Args:
            input_dir (Path): Directory path containing the images to be stitched
                (supports .jpg, .png, .tiff formats).
            output_file (Path): Path where the resulting panorama will be saved,
                including the file extension.
            mode (str): Stitching mode, can be 'full' (full pipeline with
                blending), 'auto' (automatic selection, defaults to full), or
                'collage' (minimal blending). Defaults to 'auto'.
        Returns:
            None
        Notes:
            Images are processed in a consistent order if frozen_order is True
                during parsing. The output format and quality depend on the file
                extension provided in output_file.
        """
        if self.custom_undistortion:
            camera_matrix, distortion_params = self.preproccess_undictortion(
                tiles_dir,
                cache_path,
                n_undistortions=self.n_undistortions
            )
            tmp_dir = cache_path / 'tmp'
            undistort_dir(tiles_dir, tmp_dir, camera_matrix, distortion_params)
            tiles_dir = tmp_dir

        load_matches = self.load_matches if load_matches is None else load_matches
        if load_matches:
            matches_dir = cache_path / 'matches.pkl'
            data = Serializer().load(matches_dir)
        else:
            tile_set = _tiles_parsing(tiles_dir, use_grid_info=self.use_grid_info)
            data = self.matcher.match(tile_set, use_grid_info=self.use_grid_info)

        mode = self.stitching_mode if mode is None else mode
        match mode:
            case 'save_matches':
                matches_dir = cache_path / 'matches.pkl'
                Serializer().save(data, matches_dir)
                return
            case 'full' | 'auto':
                panorama_data = self._stitch_full_pipline(data)
            case 'collage':
                panorama_data = self._stitch_collage(data)
            case 'gaincomp collage':
                panorama_data = self._stitch_compensated_collage(data)
            case 'mosaic':
                panorama_data = self._stitch_compensated_mosaic(data)
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        try:
            panorama_data.save_panorama(output_file)
        except Exception as e:
            logger.error(f"Failed to save panorama to {output_file}: {str(e)}")

        if self.custom_undistortion:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
