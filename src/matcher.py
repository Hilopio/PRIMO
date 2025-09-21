import numpy as np
import torch
import kornia.feature as KF
import gc

from src.logger import logger, log_time
from src.classes import TileSet, StitchingData, Match


class Matcher:
    def __init__(
        self, device: str, weights: str = 'outdoor',
        batch_size: int = 10, inference_size: list = [600, 400],
        use_grid_info: bool = False
    ) -> None:
        """
        Initialize the Matcher with a specified device.

        Args:
            device: The device to be used for matching operations (e.g., 'cpu' or 'cuda').
        """
        self.device = torch.device(device if device else "cpu")
        self.batch_size = batch_size
        self.inference_size = inference_size
        self.use_grid_info = use_grid_info

        if weights == 'outdoor' or weights == 'indoor':
            self.model = KF.LoFTR(pretrained=weights).to(self.device)
        else:
            self.model = KF.LoFTR(pretrained=None)
            checkpoint = torch.load(weights, map_location=device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(device)

    def prepare_pairs(self, tile_set: TileSet, use_grid_info: bool = False) -> list[tuple[int, int]]:
        n = len(tile_set.order)
        if n < 2:
            raise Exception("Not enough images to match")

        if not use_grid_info:
            ord_i, ord_j = np.triu_indices(n, k=1)
            idx_pairs = list(zip(ord_i, ord_j))
            pairs = [(tile_set.order[i], tile_set.order[j]) for i, j in idx_pairs]
        else:
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    r1, c1 = tile_set.rowcol[tile_set.order[i]]
                    r2, c2 = tile_set.rowcol[tile_set.order[j]]
                    if abs(r1 - r2) <= 2 and abs(c1 - c2) <= 2:
                        pairs.append((tile_set.order[i], tile_set.order[j]))

        return pairs

    @log_time("Matching done for", logger)
    def match(
        self, tile_set: TileSet, batch_size: int = None,
        inference_size: list = None, use_grid_info: bool = None
    ) -> StitchingData:
        """
        Match features between images to find correspondences using the LoFTR model.

        Args:
            tile_set (TileSet): A set of images to be matched, containing image data and processing order.
            batch_size (int, optional): The number of image pairs to process in each batch. Defaults to 10.

        Returns:
            StitchingData: Data object containing matching information between images, including matched points
                and confidence scores.
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        inference_size = np.array(self.inference_size if inference_size is None else inference_size)
        use_grid_info = self.use_grid_info if use_grid_info is None else use_grid_info

        pairs = self.prepare_pairs(tile_set, use_grid_info=self.use_grid_info)
        total_infer = len(pairs)
        batch_num = (total_infer - 1) // batch_size + 1

        matches: list[Match] = []
        for batch_idx in range(batch_num):
            current_pairs = pairs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            batch1: list[torch.Tensor] = []
            batch2: list[torch.Tensor] = []
            for id_i, id_j in current_pairs:
                tensor_i = tile_set.images[id_i].get_loftr_tensor(inference_size)
                tensor_j = tile_set.images[id_j].get_loftr_tensor(inference_size)
                batch1.append(tensor_i)
                batch2.append(tensor_j)

            batch1_tensor = torch.cat(batch1, dim=0).to(self.device)
            batch2_tensor = torch.cat(batch2, dim=0).to(self.device)

            try:
                with torch.inference_mode():
                    correspondences = self.model({"image0": batch1_tensor, "image1": batch2_tensor})
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"Out of memory for batch {batch_idx + 1}/{batch_num}. Consider reducing batch size.")
                raise

            batch_result = {k: v.detach().cpu() for k, v in correspondences.items()}
            del correspondences
            if 'cuda' in self.device.type:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

            for local_i, (i, j) in enumerate(current_pairs):
                id_i = tile_set.order[i]
                id_j = tile_set.order[j]

                idx = batch_result["batch_indexes"] == local_i
                if not idx.any():
                    continue

                kp0 = batch_result["keypoints0"][idx].numpy()
                kp1 = batch_result["keypoints1"][idx].numpy()
                conf = batch_result["confidence"][idx].numpy()

                xy_i = kp0 * tile_set.images[id_i].orig_size / inference_size
                xy_j = kp1 * tile_set.images[id_j].orig_size / inference_size

                matches.append(Match(id_i, id_j, xy_i, xy_j, conf))

        return StitchingData(
            tile_set=tile_set,
            matches=matches,
            reper_idx=None,
            num_dropped_images=None,
            panorama_size=None,
            canvas=None
        )
