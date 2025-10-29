# оптимизация все в одном

from typing import Optional, Literal
import numpy as np
import torch
import torch.nn as nn
import random
from torch.optim.lr_scheduler import StepLR
from kornia.geometry.calibration import undistort_points

from src.logger import logger, log_time
from src.classes import StitchingData, DistortionConfig, OptimizerConfig
from torch.nn.utils.rnn import pad_sequence


def parse_homographies(data):
    homographies = []
    for id in data.tile_set.order:
        img = data.tile_set.images[id]
        homographies.append(img.homography)
    return homographies


def set_homographies_to_data(homographies, data):
    for i, id in enumerate(data.tile_set.order):
        img = data.tile_set.images[id]
        img.homography = homographies[i]


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    if device is not None:
        if 'cuda' in device.type:
            torch.cuda.set_device(device.index)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

class Optimizer:
    def __init__(
        self,
        device: torch.device,
        data: StitchingData,
        transformation_type: Optional[Literal['affine', 'perspective']],
        undistort: bool
    ) -> None:
        self.device = device
        self.data = data
        self.reper_idx = data.reper_idx
        self.transformation_type = transformation_type
        self.undistort = undistort
        set_seed(42, device)

        # setting inliers
        (
            self.query_idx,
            self.target_idx,
            self.query_inliers_padded,
            self.target_inliers_padded
        ) = self.prepare_inliers(data.matches)
    
        # setting homographies
        homographies = parse_homographies(data)
        self.n_tiles = len(homographies)
        match transformation_type:
            case 'affine':
                self.prepare_homography_params(homographies)  # set a_params, b_params
            case 'perspective':
                self.prepare_homography_params(homographies)
            case _:
                raise ValueError(f"Unknown transformation type: {transformation_type}")
                
        # setting distortion params
        if undistort:
            self.prepare_distortion_params(DistortionConfig())
        


    def prepare_inliers(self, inliers):
        query_idx = []
        target_idx = []
        query_inliers = []
        target_inliers = []
        for inlier in inliers:
            query_idx.append(inlier.i)
            target_idx.append(inlier.j)
            query_inlier = torch.from_numpy(inlier.xy_i)
            homogenuous_query = torch.cat((query_inlier, torch.ones((query_inlier.shape[0], 1))), dim=1)
            query_inliers.append(homogenuous_query)
            target_inlier = torch.from_numpy(inlier.xy_j)
            homogenuous_target = torch.cat((target_inlier, torch.ones((target_inlier.shape[0], 1))), dim=1)
            target_inliers.append(homogenuous_target)

        query_inliers_padded = pad_sequence(query_inliers, batch_first=True, padding_value=0.0)
        target_inliers_padded = pad_sequence(target_inliers, batch_first=True, padding_value=0.0)
        query_inliers_padded = query_inliers_padded.permute(0, 2, 1)
        target_inliers_padded = target_inliers_padded.permute(0, 2, 1)
        
        assert query_inliers_padded.shape[0] == len(inliers)
        assert query_inliers_padded.shape[1] == 3
        assert query_inliers_padded.shape == target_inliers_padded.shape, (query_inliers_padded.shape, target_inliers_padded.shape)

        # inliers_padded.shape = (num_img_pairs, 3, max_inliers)
        return query_idx, target_idx, query_inliers_padded, target_inliers_padded

    def prepare_distortion_params(self, cfg: DistortionConfig):
        f = 1e4
        some_id = self.data.tile_set.order[0]
        w, h = self.data.tile_set.images[some_id].orig_size
        cx, cy = w // 2, h // 2

        self.log_f = nn.Parameter(
            torch.tensor(
                [torch.log(torch.tensor(f))],
                device=self.device,
                dtype=torch.float32,
                requires_grad=True
            )
        )

        self.c_xy = nn.Parameter(
            torch.tensor(
                [cx, cy], device=self.device, dtype=torch.float32,
                requires_grad=not cfg.freeze_principal_point
            )
        )

        self.k1 = nn.Parameter(torch.tensor([cfg.k1], device=self.device, dtype=torch.float32, requires_grad=True))
        self.k2 = nn.Parameter(torch.tensor([cfg.k2], device=self.device, dtype=torch.float32, requires_grad=True))
        self.k3 = nn.Parameter(torch.tensor([cfg.k3], device=self.device, dtype=torch.float32, requires_grad=True))

        self.p = nn.Parameter(
            torch.tensor(
                [cfg.p1, cfg.p2], device=self.device, dtype=torch.float32,
                requires_grad=not cfg.freeze_tangential
            )
        )
    
    def prepare_homography_params(self, homographies):
        assert np.allclose(homographies[self.reper_idx], np.eye(3))

        all_homographies = np.array(homographies)
        all_homographies = np.concatenate((
            all_homographies[:self.reper_idx], all_homographies[self.reper_idx + 1:]
        ), axis=0)

        assert all_homographies.shape == (self.n_tiles - 1, 3, 3)
        a_params = all_homographies[:, :2, :2].reshape(self.n_tiles - 1, 4)
        b_params = all_homographies[:, :2, 2]
        assert b_params.shape == (self.n_tiles - 1, 2)

        self.a_params = torch.nn.Parameter(torch.from_numpy(a_params).to(self.device))
        self.b_params = torch.nn.Parameter(torch.from_numpy(b_params).to(self.device))

        if self.transformation_type == 'perspective':
            c_params = all_homographies[:, 2, :2]
            assert c_params.shape == (self.n_tiles - 1, 2)
            self.c_params = torch.nn.Parameter(torch.from_numpy(c_params).to(self.device))


    @property
    def f(self):
        """Вычисляет f из логарифмического параметра: f = exp(log_f)"""
        return torch.exp(self.log_f)


    def get_camera_matrix_batch(self, batch_size=1) -> torch.Tensor:

        one = torch.ones(batch_size, device=self.device, dtype=torch.float32)
        f_batch = self.f.expand(batch_size)
        cx_batch = self.c_xy[0].expand(batch_size)
        cy_batch = self.c_xy[1].expand(batch_size)

        camera_matrix = torch.zeros(batch_size, 3, 3, device=self.device, dtype=torch.float32)
        camera_matrix[:, 0, 0] = f_batch
        camera_matrix[:, 1, 1] = f_batch
        camera_matrix[:, 0, 2] = cx_batch
        camera_matrix[:, 1, 2] = cy_batch
        camera_matrix[:, 2, 2] = one

        return camera_matrix

    def get_distortion_params_batch(self, batch_size=1) -> torch.Tensor:
        """Батчевая версия distortion parameters"""
        dist_coeffs = torch.zeros((batch_size, 14), device=self.device, dtype=torch.float32)
        dist_coeffs[:, 0] = self.k1.expand(batch_size)
        dist_coeffs[:, 1] = self.k2.expand(batch_size)
        dist_coeffs[:, 2] = self.p[0].expand(batch_size)
        dist_coeffs[:, 3] = self.p[1].expand(batch_size)
        dist_coeffs[:, 4] = self.k3.expand(batch_size)
        return dist_coeffs

    
    def get_homographies(self):
        a = self.a_params.view(self.n_tiles - 1, 2, 2)
        b = self.b_params.view(self.n_tiles - 1, 2, 1)

        match self.transformation_type:
            case 'affine':
                c = torch.zeros((self.n_tiles - 1, 1, 2), device=self.device, dtype=torch.float32)
            case 'perspective':
                c = self.c_params.view(self.n_tiles - 1, 1, 2)
            case _:
                raise ValueError(f"Unknown transformation type: {self.transformation_type}")

        d = torch.ones((self.n_tiles - 1, 1, 1), device=self.device, dtype=torch.float32)
        ab = torch.cat((a, b), dim=2)
        cd = torch.cat((c, d), dim=2)
        homographies = torch.cat((ab, cd), dim=1)
        homographies = torch.cat((
            homographies[:self.reper_idx],
            torch.eye(3, device=self.device, dtype=torch.float32).view(1, 3, 3),
            homographies[self.reper_idx:]
        ))
        assert homographies.shape == (self.n_tiles, 3, 3)
        return homographies


    def udistort_inliers(self, padded_inliers):

        assert len(padded_inliers.shape) == 3
        assert padded_inliers.shape[1] == 3
    
        batch_size = padded_inliers.shape[0]
        camera_matrix = self.get_camera_matrix_batch(batch_size)
        dist_coeffs = self.get_distortion_params_batch(batch_size)

        homogenuous = padded_inliers[:, 2:3]
        corrected_inliers = undistort_points(
            points=padded_inliers[:, :2].permute(0, 2, 1), # (batch_size, n_points, 2)
            K=camera_matrix,  # (batch_size, 3, 3)
            dist=dist_coeffs, # (batch_size, 14)
            num_iters=5
        )
        corrected_inliers = torch.cat((corrected_inliers.permute(0, 2, 1), homogenuous), dim=1)

        assert len(corrected_inliers.shape) == 3
        assert corrected_inliers.shape[1] == 3
        return corrected_inliers  # (batch_size, 3, max_inliers)

    def compute_reproj_mse(self, undistort=None) -> torch.Tensor:

        undistort = self.undistort if undistort is None else undistort

        if undistort:
            corrected_query_inliers = self.udistort_inliers(self.query_inliers_padded)
            corrected_target_inliers = self.udistort_inliers(self.target_inliers_padded)
        else:
            corrected_query_inliers = self.query_inliers_padded
            corrected_target_inliers = self.target_inliers_padded

        homographies = self.get_homographies()
        query_homographies = homographies[self.query_idx]
        target_homographies = homographies[self.target_idx]

        mask = self.query_inliers_padded[:, 2] == 0

        query_inliers_proj = torch.bmm(query_homographies, corrected_query_inliers)
        target_inliers_proj = torch.bmm(target_homographies, corrected_target_inliers)

        query_inliers_proj[:, 2] = query_inliers_proj[:, 2] + mask.float()
        target_inliers_proj[:, 2] = target_inliers_proj[:, 2] + mask.float()

        query_inliers_proj = query_inliers_proj / query_inliers_proj[:, 2:3]
        target_inliers_proj = target_inliers_proj / target_inliers_proj[:, 2:3]

        n_valid_inliers = mask.numel() - mask.sum()
        reproj_mse = (query_inliers_proj - target_inliers_proj) ** 2
        with torch.no_grad():
            assert reproj_mse[:, 2].sum() == 0
        reproj_mse = reproj_mse.sum() / n_valid_inliers


        return reproj_mse

    def get_rmse(self, undistort=None):
        with torch.no_grad():
            re = self.compute_reproj_mse(undistort)
            return np.sqrt(re.item())

    @log_time("Bundle adjustment done for", logger)
    def bundle_adjustment(self) -> float:
        dist_cfg = DistortionConfig()
        opt_cfg = OptimizerConfig()

        # setting homographies optimizer
        homographies_params = [
            {'params': [self.a_params], 'lr': opt_cfg.a_params_lr},
            {'params': [self.b_params], 'lr': opt_cfg.b_params_lr},
        ]
        if self.transformation_type == 'projective':
            homographies_params.append({'params': [self.c_params], 'lr': 1e-6})
        homographies_optimizer = torch.optim.Adam(homographies_params, betas=(0.9, 0.999), eps=1e-8)
        homographies_scheduler = StepLR(homographies_optimizer, step_size=1000, gamma=opt_cfg.h_gamma)

        # setting distortion optimizer
        if self.undistort:
            distortion_params = [
                {'params': [self.log_f], 'lr': opt_cfg.lr_log_f},  # Параметр для log(f)
                {'params': [self.k1], 'lr': opt_cfg.lr_k1},
                {'params': [self.k2], 'lr': opt_cfg.lr_k2},
                {'params': [self.k3], 'lr': opt_cfg.lr_k3},
            ]

            if not dist_cfg.freeze_principal_point:
                distortion_params.append({'params': [self.c_xy], 'lr': opt_cfg.lr_c})

            if not dist_cfg.freeze_tangential:
                distortion_params.append({'params': [self.p], 'lr': opt_cfg.lr_p})

            distortion_optimizer = torch.optim.Adam(distortion_params, betas=(0.9, 0.999), eps=1e-8)
            distortion_scheduler = StepLR(distortion_optimizer, step_size=1000, gamma=opt_cfg.d_gamma)

        # early stopping params
        best_loss = float('inf')
        patience = 50
        tol = 1e-2
        no_improve_count = 0

        # optimization loop
        for iteration in range(opt_cfg.max_iter):
            reproj_error = self.compute_reproj_mse()
            reproj_error.backward()
            homographies_optimizer.step()
            homographies_scheduler.step()
            homographies_optimizer.zero_grad()

            if self.undistort:
                distortion_optimizer.step()
                distortion_scheduler.step()
                distortion_optimizer.zero_grad()

            current_loss = reproj_error.item()
            if best_loss - current_loss > tol:
                best_loss = current_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= patience:
                logger.debug(f"Early stopping at iteration {iteration}")
                break

        with torch.no_grad():
            homographies = self.get_homographies().detach().cpu().numpy()
            set_homographies_to_data(homographies, self.data)

            if self.undistort:
                self.data.camera_matrix = self.get_camera_matrix_batch(1)[0].detach().cpu().numpy()
                self.data.distortion_params = self.get_distortion_params_batch(1)[0, :5].detach().cpu().numpy()

        return self.data

