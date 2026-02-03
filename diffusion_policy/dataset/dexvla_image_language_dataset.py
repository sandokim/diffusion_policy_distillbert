from typing import Dict, List, Optional

import glob
import os

import h5py
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class DexVLAMultiEpisodeImageLanguageDataset(BaseImageDataset):
    """
    Multi-episode dataset for DexVLA-style hdf5 files, e.g.:

    /home/hyeseong/diffusion_policy/dexvla_example_data/episode_47.hdf5

    Expected structure per episode file:
        action:              (T, 14)
        base_action:         (T, 2)          # currently unused
        language_raw:        (1,)            # e.g. "Cook rice."
        observations/effort: (T, 14)
        observations/images/cam_high: (T, H, W, 3)
        observations/qpos:   (T, 14)
        observations/qvel:   (T, 14)
        sub_reason_distilbert: (T, 1, 768) or (T, 768)

    For each episode we use:
        obs.image      <- cam_high
        obs.qpos       <- qpos
        action         <- action
        language_emb   <- sub_reason_distilbert squeezed to (T, 768)
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        file_pattern: str = "episode_*.hdf5",
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
    ):
        super().__init__()

        # Build in-memory replay buffer from multiple hdf5 episodes for one task
        replay_buffer = ReplayBuffer.create_empty_numpy()

        # Collect episode file paths
        episode_paths: List[str] = []
        if dataset_path is not None:
            episode_paths.append(os.path.expanduser(dataset_path))
        if dataset_dir is not None:
            pattern = os.path.join(os.path.expanduser(dataset_dir), file_pattern)
            episode_paths.extend(sorted(glob.glob(pattern)))

        if len(episode_paths) == 0:
            raise ValueError(
                "DexVLAImageLanguageDataset: no episodes found. "
                "Set dataset_path to a file or dataset_dir to a directory "
                f"containing '{file_pattern}'."
            )

        # Load all episodes into a single ReplayBuffer
        for ep_path in episode_paths:
            with h5py.File(ep_path, "r") as f:
                # time dimension
                actions = f["action"][:].astype(np.float32)  # (T, 14)
                cam_high = f["observations"]["images"]["cam_high"][:]  # (T, H, W, 3), uint8
                qpos = f["observations"]["qpos"][:].astype(np.float32)  # (T, 14)

                # precomputed DistilBERT embeddings per (sub-)step
                # stored at root as "sub_reason_distilbert"
                sub_reason = f["sub_reason_distilbert"][:]  # (T, 1, 768) or (T, 768)
                # squeeze middle dim if present -> (T, 768)
                if sub_reason.ndim == 3:
                    sub_reason = sub_reason[:, 0, :]
                language_emb = sub_reason.astype(np.float32)

                episode_data = {
                    "action": actions,
                    "image": cam_high,
                    "qpos": qpos,
                    "language_emb": language_emb,
                }

                replay_buffer.add_episode(episode_data)

        self.replay_buffer = replay_buffer

        # single-episode val split (can keep val_ratio=0.0 to avoid validation)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = DexVLAImageLanguageDataset.__new__(DexVLAImageLanguageDataset)
        # shallow copy of internal state
        val_set.__dict__ = self.__dict__.copy()
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        """
        Normalize action and low-dim qpos similar to PushTImageDataset.
        Image is normalized to [0,1] by get_image_range_normalizer.
        """
        data = {
            "action": self.replay_buffer["action"],
            "qpos": self.replay_buffer["qpos"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # image: T, H, W, C -> T, C, H, W, float32 in [0,1]
        image = np.moveaxis(sample["image"], -1, 1).astype(np.float32) / 255.0
        qpos = sample["qpos"].astype(np.float32)
        action = sample["action"].astype(np.float32)
        language_emb = sample["language_emb"].astype(np.float32)  # (T, 768)

        data = {
            "obs": {
                "image": image,
                "qpos": qpos,
            },
            "action": action,
            # language embedding per timestep, used by FiLM in the policy
            "language_emb": language_emb,
        }
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class DexVLAImageLanguageDataset(DexVLAMultiEpisodeImageLanguageDataset):
    """
    Backwards-compatible alias.

    Conceptually this is a *task-level* dataset that aggregates
    multiple demonstration episodes (multiple hdf5 files) for a
    single task, matching the standard Diffusion Policy setting.
    """
    pass

