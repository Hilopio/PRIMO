from src.classes import StitchingData
import numpy as np
# from scipy.optimize import least_squares
from src.logger import logger, log_time
from abc import ABC, abstractmethod
import torch
from torch.nn.utils.rnn import pad_sequence


class Optimizer(ABC):

    def __init__(self, data: StitchingData):
        self.data = data
        self.inliers = data.matches
        self.reper_idx = data.reper_idx
        self.homographies = []
        for id in data.tile_set.order:
            img = data.tile_set.images[id]
            self.homographies.append(img.homography)
        self.n_tiles = len(self.homographies)
        self.device = torch.device('cpu')

        a_params, b_params = self.set_params(self.homographies)
        self.a_params = torch.nn.Parameter(torch.from_numpy(a_params).to(self.device))
        self.b_params = torch.nn.Parameter(torch.from_numpy(b_params).to(self.device))

        query_idx, target_idx, query_inliers_padded, target_inliers_padded = self.preproc_inliers()

        self.query_idx = query_idx
        self.target_idx = target_idx
        self.query_inliers_padded = query_inliers_padded
        self.target_inliers_padded = target_inliers_padded
    

    def set_params(self, homographies):
        assert np.allclose(homographies[self.reper_idx], np.eye(3))

        all_homographies = np.array(homographies)
        all_homographies = np.concatenate((
            all_homographies[:self.reper_idx], all_homographies[self.reper_idx + 1:]
        ), axis=0)

        assert all_homographies.shape == (self.n_tiles - 1, 3, 3)
        a_params = all_homographies[:, :2, :2].reshape(self.n_tiles - 1, 4)
        b_params = all_homographies[:, :2, 2]
        assert b_params.shape == (self.n_tiles - 1, 2)

        return a_params, b_params
    
    def get_homographies(self):
        a = self.a_params.view(self.n_tiles - 1, 2, 2)
        b = self.b_params.view(self.n_tiles - 1, 2, 1)

        c = torch.zeros((self.n_tiles - 1, 1, 2), device=self.device, dtype=torch.float32)
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
    
    def preproc_inliers(self):
        query_idx = []
        target_idx = []
        query_inliers = []
        target_inliers = []
        # valid_inliers = []
        for inlier in self.inliers:
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
        
        assert query_inliers_padded.shape[0] == len(self.inliers)
        assert query_inliers_padded.shape[1] == 3
        assert query_inliers_padded.shape == target_inliers_padded.shape, (query_inliers_padded.shape, target_inliers_padded.shape)
        return query_idx, target_idx, query_inliers_padded, target_inliers_padded
        
    def project(self, homographies, inliers):
        '''
        homographies: (n_img_pairs, 3, 3)
        inliers: (n_img_pairs, 3, max_inliers)


        returns: (n_img_pairs, 3, max_inliers)
        '''
        assert homographies.shape[1] == 3, f'{homographies.shape=}'
        assert homographies.shape[2] == 3, f'{homographies.shape=}'
        assert homographies.shape[0] == inliers.shape[0], (f'{homographies.shape=}', f'{inliers.shape=}')
        assert inliers.shape[1] == 3, f'{inliers.shape=}'
        return torch.bmm(homographies, inliers)

    def compute_re(self):
        homographies = self.get_homographies()
        query_homographies = homographies[self.query_idx]
        target_homographies = homographies[self.target_idx]
        query_inliers_proj = self.project(query_homographies, self.query_inliers_padded)  # (n_img_pairs, 3, max_inliers)
        target_inliers_proj = self.project(target_homographies, self.target_inliers_padded)  # (n_img_pairs, 3, max_inliers)

        mask = query_inliers_proj[:, 2] == 0
        query_inliers_proj[:, 2] = query_inliers_proj[:, 2] + mask.float()
        target_inliers_proj[:, 2] = target_inliers_proj[:, 2] + mask.float()

        query_inliers_proj = query_inliers_proj / query_inliers_proj[:, 2:3]
        target_inliers_proj = target_inliers_proj / target_inliers_proj[:, 2:3]

        n_valid_inliers = mask.numel() - mask.sum()
        re = (query_inliers_proj - target_inliers_proj) ** 2
        with torch.no_grad():
            assert re[:, 2].sum() == 0
        re = re.sum() / n_valid_inliers

        return re
    
    def get_rmse(self):
        with torch.no_grad():
            re = self.compute_re()
            return np.sqrt(re.item())
    
    def save_result(self):
        with torch.no_grad():
            homographies = self.get_homographies().detach().cpu().numpy()
            for i, id in enumerate(self.data.tile_set.order):
                img = self.data.tile_set.images[id]
                img.homography = homographies[i]

    @log_time("Bundle adjustment done for", logger)
    def bundle_adjustment(self):
        print("Starting bundle adjustment")
        optimizer = torch.optim.Adam([
            {'params': self.a_params, 'lr': 1e-3},
            {'params': self.b_params, 'lr': 1e-1}
        ])

        n_iterations = 1000
        for iter in range(n_iterations):
            optimizer.zero_grad()
            re = self.compute_re()
            re.backward()
            optimizer.step()

            # reprojection_error = np.sqrt(re.item())
            # if iter % 100 == 0:
            #     logger.info(f"{reprojection_error=} on iteration {iter}")

        self.save_result()
        return self.data