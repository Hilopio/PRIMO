from src.classes import StitchingData, TileSet, AlignConfig
from src.optimizer import Optimizer
from src.distortion_optimizer import DistortionOptimizer, DistortionOptimizerBase
from src.align_functions import matches_alignment, translate_and_add_panorama_size

from src.logger import logger
import torch


def _simple_align_iteration(data: StitchingData, cfg: AlignConfig) -> StitchingData:

    temp_data = matches_alignment(
        data.copy(), cfg.transformation_type, cfg.confidence_threshold, cfg.min_inliers_for_accept,
        cfg.max_used_inliers, cfg.relative_reproj_threshold, cfg.max_recenter_iterations
    )

    optimizer = Optimizer(cfg.transformation_type, temp_data)
    vec = optimizer.homography_to_vec(optimizer.homographies)
    error = optimizer.reprojection_error(vec)
    error = (error ** 2).mean() ** 0.5
    logger.debug(f"Initial error: {error}")

    if cfg.use_bundle_adjustment:
        temp_data = optimizer.bundle_adjustment()
        vec = optimizer.homography_to_vec(optimizer.homographies)
        error = optimizer.reprojection_error(vec)
        error = (error ** 2).mean() ** 0.5
        logger.debug(f"Optimized error: {error}")  # переписать красиво подсчет ошибки

    return error, temp_data


def _adaptive_sycle(data: StitchingData, cfg: AlignConfig) -> StitchingData:
    attempts = {}
    current_min_inliers = cfg.min_inliers_for_accept
    if cfg.adaptive:
        steps = cfg.min_inliers_refinement_steps
    else:
        steps = []
    max_attempts = len(steps) + 1

    error_threshold = (
        cfg.optimized_error_threshold
        if cfg.use_bundle_adjustment
        else cfg.initial_error_threshold
    )
    for attempt in range(max_attempts):
        try:
            current_cfg = cfg.copy()
            current_cfg.min_inliers_for_accept = current_min_inliers
            error, temp_data = _simple_align_iteration(data, current_cfg)

            has_dropped = temp_data.num_dropped_images > 0
            has_high_error = error > error_threshold

            # retry log warning
            if attempt == 0 and (has_dropped or has_high_error):
                log = ''
                if has_dropped:
                    log += f"Probably missing {temp_data.num_dropped_images} images. Panorama may be not complete. "

                if has_high_error:
                    log += "Probably contains false connections. Panorama may be distorted. "

                log += 'Using adaptive strictness.'
                logger.warning(log)

            if not has_dropped and not has_high_error:
                attempts = {current_min_inliers: (temp_data.num_dropped_images, error, temp_data)}
                return attempts

            attempts[current_min_inliers] = (temp_data.num_dropped_images, error, temp_data)

            if attempt == max_attempts - 1:
                break

            step = steps[attempt]

            if has_high_error:
                current_min_inliers += step
            else:
                current_min_inliers -= step

            # Clamp min_inliers to reasonable bounds
            current_min_inliers = max(5, current_min_inliers)
        except Exception as e:
            logger.error(f"Alignment attempt with min_inliers={current_min_inliers} failed: {str(e)}")
    return attempts


def log_choosen(attempts, best_key):
    if len(attempts.keys()) > 0:
        logger.debug(
            f"Selected best attempt with min_inliers={best_key}, \
            dropped={attempts[best_key][0]}, \
            error={attempts[best_key][1]}"
        )
    else:
        raise Exception("No valid attempts")


def _choose_best_attempt(
    cfg: AlignConfig,
    attempts: dict[int, tuple[int, float, StitchingData | DistortionOptimizerBase]]
) -> StitchingData | DistortionOptimizerBase:

    error_threshold = (
        cfg.optimized_error_threshold
        if cfg.use_bundle_adjustment
        else cfg.initial_error_threshold
    )
    # Rule 1: Filter attempts with error below threshold
    valid_attempts = {
        k: v for k, v in attempts.items() if v[1] <= error_threshold
    }

    if valid_attempts:
        # Choose attempt with minimum dropped tiles; if equal, max min_inliers
        best_key = min(
            valid_attempts,
            key=lambda k: (valid_attempts[k][0], -k)  # Minimize dropped, maximize min_inliers
        )
        best_data = valid_attempts[best_key][2]
        log_choosen(valid_attempts, best_key)
        return best_data

    # Rule 2: If all errors >= threshold, select where dropped <= MAX_DROPPED_TILES and max min_inliers
    valid_attempts = {
        k: v for k, v in attempts.items() if v[0] <= cfg.max_dropped_tiles
    }
    if valid_attempts:
        best_key = max(valid_attempts, key=lambda k: k)  # Maximize min_inliers
        best_data = valid_attempts[best_key][2]
        log_choosen(valid_attempts, best_key)
        return best_data

    # Rule 3: Fallback to min_inliers=default
    best_key = cfg.min_inliers_for_accept
    best_data = attempts[best_key][2]
    log_choosen(attempts, best_key)
    return best_data


def _align(data: StitchingData, cfg: AlignConfig) -> StitchingData:
    try:
        attempts = _adaptive_sycle(data, cfg)
    except Exception as e:
        logger.error(f"Adaptive alignment failed: {str(e)}")

    try:
        best_data = _choose_best_attempt(cfg, attempts)
    except Exception as e:
        logger.error(f"Best attempt selection failed: {str(e)}")
    best_data = translate_and_add_panorama_size(best_data)
    return best_data


def _simple_CUBA_iteration(data: StitchingData, cfg: AlignConfig):

    lr_log_f: float = 1e-2
    lr_c: float = 3.585612610345396
    lr_k1: float = 0.07556810141274425
    lr_k2: float = 0.001260466458564947
    lr_k3: float = 5.727904470799619e-07
    lr_p: float = 0.0003795853142670637
    h_gamma: float = 0.9
    d_gamma: float = 0.9

    f = 1e4
    some_id = data.tile_set.order[0]
    w, h = data.tile_set.images[some_id].orig_size
    cx, cy = w // 2, h // 2

    device = torch.device('cpu')

    data = matches_alignment(
        data.copy(), cfg.transformation_type, cfg.confidence_threshold,
        cfg.min_inliers_for_accept, cfg.max_used_inliers,
        cfg.relative_reproj_threshold, cfg.max_recenter_iterations
    )

    d_optimizer = DistortionOptimizer('affine', device, data, f=f, cx=cx, cy=cy)
    try:
        error = d_optimizer.bundle_adjustment(
            # lr_f=lr_f,
            lr_log_f=lr_log_f,
            lr_c=lr_c,
            lr_k1=lr_k1,
            lr_k2=lr_k2,
            lr_k3=lr_k3,
            lr_p=lr_p,
            h_gamma=h_gamma,
            d_gamma=d_gamma,
            max_iter=2000,
            verbose='core'
        )
    except Exception as e:
        logger.error(f"Affine bundle adjustment failed: {e}")
    return error, data, d_optimizer


def _adaptive_sycle_CUBA(data: StitchingData, cfg: AlignConfig):
    attempts = {}
    current_min_inliers = cfg.min_inliers_for_accept
    if cfg.adaptive:
        steps = cfg.min_inliers_refinement_steps
    else:
        steps = []
    max_attempts = len(steps) + 1

    error_threshold = (
        cfg.optimized_error_threshold
        if cfg.use_bundle_adjustment
        else cfg.initial_error_threshold
    )
    for attempt in range(max_attempts):
        try:
            current_cfg = cfg.copy()
            current_cfg.min_inliers_for_accept = current_min_inliers
            error, temp_data, optimizer = _simple_CUBA_iteration(data, current_cfg)

            has_dropped = temp_data.num_dropped_images > 0
            has_high_error = error > error_threshold

            # retry log warning
            if attempt == 0 and (has_dropped or has_high_error):
                log = ''
                if has_dropped:
                    log += f"Probably missing {temp_data.num_dropped_images} images. Panorama may be not complete. "

                if has_high_error:
                    log += "Probably contains false connections. Panorama may be distorted. "

                log += 'Using adaptive strictness.'
                logger.warning(log)

            if not has_dropped and not has_high_error:
                attempts = {current_min_inliers: (temp_data.num_dropped_images, error, optimizer)}
                return attempts

            attempts[current_min_inliers] = (temp_data.num_dropped_images, error, optimizer)
            if attempt == max_attempts - 1:
                break

            step = steps[attempt]

            if has_high_error:
                current_min_inliers += step
            else:
                current_min_inliers -= step

            # Clamp min_inliers to reasonable bounds
            current_min_inliers = max(5, current_min_inliers)
        except Exception as e:
            logger.error(f"Alignment attempt with min_inliers={current_min_inliers} failed: {str(e)}")
    return attempts


def _CUBA(  # Custom Undistortion Bundle Adjustment
    data: StitchingData,
    cfg: AlignConfig
) -> TileSet:

    try:
        attempts = _adaptive_sycle_CUBA(data, cfg)
    except Exception as e:
        logger.error(f"Adaptive alignment failed: {str(e)}")

    try:
        optimizer = _choose_best_attempt(cfg, attempts)
    except Exception as e:
        logger.error(f"Best attempt selection failed: {str(e)}")

    affine_cm = optimizer.get_camera_matrix_batch().squeeze().cpu().detach().numpy()
    affine_dp = optimizer.get_distortion_params_batch().cpu().detach().numpy()[:, :5]
    return affine_cm, affine_dp
