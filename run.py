from src.matcher import Matcher
from src.stitcher import Stitcher
from src.runner import Runner
from src.classes import AlignConfig
from src.logger import logger

import resource
import os
import hydra
from omegaconf import DictConfig

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    matcher = Matcher(
        device=cfg.match.device,
        weights=cfg.match.matcher_weights,
        batch_size=cfg.match.batch_size,
        inference_size=cfg.match.inference_size,
        use_grid_info=cfg.use_grid_info
    )

    align_cfg = AlignConfig(**cfg.align)

    stitcher = Stitcher(
        matcher=matcher,
        load_matches=cfg.load_matches,
        align_cfg=align_cfg,

        use_gain_comp=cfg.compose.use_gain_comp,
        use_graphcut=cfg.compose.use_graphcut,
        use_blending=cfg.compose.use_blending,

        save_mean_color=cfg.gain_comp.save_mean_color,

        coarse_scale=cfg.graphcut.coarse_scale,
        fine_scale=cfg.graphcut.fine_scale,
        lane_width=cfg.graphcut.lane_width,

        n_levels=cfg.blending.n_levels,

        detailed_log=cfg.log.detailed_log,

        draw_inliers=cfg.vizualization.draw_inliers,
        draw_connections=cfg.vizualization.draw_connections,

        use_grid_info=cfg.use_grid_info,
        custom_undistortion=cfg.custom_undistortion,
        n_undistortions=cfg.n_undistortions,
        stitching_mode=cfg.stitching_mode
    )

    max_heap_size = 100 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_DATA, (max_heap_size, max_heap_size))

    runner = Runner()
    logger.info(f"Processing collection with params: \n{align_cfg}")
    try:
        runner.process_collection(
            stitcher=stitcher,
            tiles_metadir=cfg.dirs.tiles_dir,
            cache_metadir=cfg.dirs.cache_dir,
            output_metadir=cfg.dirs.panoramas_dir
        )
    except MemoryError:
        logger.error("MemoryError: Not enough memory to process collection (100 MB)")


if __name__ == "__main__":
    main()
