from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from src.dataset.types import BatchedExample
from src.model.decoder.decoder import DecoderOutput
from src.model.types import Gaussians
from .loss import Loss
from typing import Generic, TypeVar
from dataclasses import fields
import torch.nn.functional as F
import sys
import os

try:
    from pytorch3d.loss import chamfer_distance
except ImportError:
    chamfer_distance = None  # pytorch3d not available - only needed for training
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.loss.depth_anything.dpt import DepthAnything
from src.misc.utils import vis_depth_map

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


@dataclass
class LossChamferDistanceCfg:
    weight: float
    down_sample_ratio: float
    sigma_image: float | None


@dataclass
class LossChamferDistanceCfgWrapper:
    chamfer_distance: LossChamferDistanceCfg

class LossChamferDistance(Loss[LossChamferDistanceCfg, LossChamferDistanceCfgWrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)
        
        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict,
        global_step: int,
    ) -> Float[Tensor, ""]:
        if chamfer_distance is None:
            raise ImportError(
                "pytorch3d is required for LossChamferDistance. "
                "Please install it with: pip install 'git+https://github.com/facebookresearch/pytorch3d.git'"
            )
        # Scale the depth between the near and far planes.
        b, v, h, w, _ = depth_dict['distill_infos']['pts_all'].shape
        pred_pts = depth_dict['distill_infos']['pts_all'].flatten(0, 1)

        conf_mask = depth_dict['distill_infos']['conf_mask']
        gaussian_meas = gaussians.means

        pred_pts = pred_pts.view(b, v, h, w, -1)
        conf_mask = conf_mask.view(b, v, h, w)

        pts_mask = torch.abs(gaussian_meas[..., -1]) < 1e2 # 
        # conf_mask = conf_mask & pts_mask
        
        cd_losses = 0.0
        for b_idx in range(b):
            batch_pts, batch_conf, batch_gaussian = pred_pts[b_idx], conf_mask[b_idx], gaussian_meas[b_idx][pts_mask[b_idx]]
            batch_pts = batch_pts[batch_conf]
            batch_pts = batch_pts[torch.randperm(batch_pts.shape[0])[:int(batch_pts.shape[0] * self.cfg.down_sample_ratio)]]
            batch_gaussian = batch_gaussian[torch.randperm(batch_gaussian.shape[0])[:int(batch_gaussian.shape[0] * self.cfg.down_sample_ratio)]]
            cd_loss = chamfer_distance(batch_pts.unsqueeze(0), batch_gaussian.unsqueeze(0))[0]
            cd_losses = cd_losses + cd_loss
        return self.cfg.weight * torch.nan_to_num(cd_losses / b, nan=0.0)