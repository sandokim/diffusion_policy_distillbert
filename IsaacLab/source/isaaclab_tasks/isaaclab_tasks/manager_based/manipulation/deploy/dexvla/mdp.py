from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done_cook_rice(
    env: ManagerBasedRLEnv,
    cooker_cfg: SceneEntityCfg = SceneEntityCfg("rice_cooker"),
    min_x: float = 0.25,
    max_x: float = 0.55,
    min_y: float = 0.35,
    max_y: float = 0.75,
    max_height: float = 1.05,
) -> torch.Tensor:
    """Simple success condition: cooker placed within a target region."""
    cooker: RigidObject = env.scene[cooker_cfg.name]
    pos = cooker.data.root_pos_w - env.scene.env_origins

    done = pos[:, 0] > min_x
    done = torch.logical_and(done, pos[:, 0] < max_x)
    done = torch.logical_and(done, pos[:, 1] > min_y)
    done = torch.logical_and(done, pos[:, 1] < max_y)
    done = torch.logical_and(done, pos[:, 2] < max_height)
    return done


