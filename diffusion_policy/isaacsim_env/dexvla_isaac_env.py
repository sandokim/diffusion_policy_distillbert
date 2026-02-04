from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Callable, Any

import numpy as np
import torch


class DexVLAIsaacEnv:
    """
    Isaac Lab manager-based env wrapper for DexVLA evaluation.

    This wrapper keeps the Diffusion Policy side simple:
    - reset() -> obs dict
    - step(action) -> (obs, reward, done, info)

    It expects the underlying Isaac Lab env to be a ManagerBasedRLEnv.
    """

    def __init__(
        self,
        task_cfg_class_path: str,
        isaaclab_source: Optional[str] = None,
        headless: bool = True,
        enable_cameras: bool = True,
        device: str = "cuda:0",
        episode_idx: int = 0,
        enable_render: bool = True,
        render_fps: int = 10,
        crf: int = 22,
        output_dir: Optional[str] = None,
        obs_adapter: Optional[Callable[[Any], Dict[str, np.ndarray]]] = None,
        action_adapter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Args:
            task_cfg_class_path: Python path to the IsaacLab manager-based env config class.
                Example:
                "isaaclab_tasks.manager_based.manipulation.deploy.reach.reach_env_cfg.ReachEnvCfg"
            isaaclab_source: Path to IsaacLab source directory (e.g. /home/hyeseong/IsaacLab/source).
                If provided, it is appended to sys.path for local imports.
            headless: Whether to run Isaac Sim in headless mode.
            enable_cameras: Whether to enable cameras (required for image observations).
            device: Isaac Sim device string, e.g. "cuda:0".
            episode_idx: Episode index (for deterministic seeding or logging).
            enable_render: Whether to record/enable rendering.
            render_fps: FPS for video recording (if implemented on the Isaac side).
            crf: Video CRF (if recording).
            output_dir: Output directory for video recording.
            obs_adapter: Optional function to map raw Isaac observations to DP obs dict.
            action_adapter: Optional function to map DP action to Isaac env action.
        """
        if isaaclab_source is not None:
            isaaclab_source = os.path.expanduser(isaaclab_source)
            # Ensure the provided IsaacLab source takes priority over any existing ones.
            sys.path = [p for p in sys.path if "IsaacLab/source" not in p]
            # Add IsaacLab extensions explicitly so imports work without /home/hyeseong/IsaacLab
            candidate_paths = [
                isaaclab_source,
                os.path.join(isaaclab_source, "isaaclab"),
                os.path.join(isaaclab_source, "isaaclab_tasks"),
                os.path.join(isaaclab_source, "isaaclab_assets"),
            ]
            for p in reversed(candidate_paths):
                if p not in sys.path:
                    sys.path.insert(0, p)
            print(f"[DexVLAIsaacEnv] isaaclab_source: {isaaclab_source}")

        # Lazy import to avoid hard dependency in non-Isaac environments.
        # IMPORTANT: AppLauncher must be instantiated before importing other Isaac modules.
        from isaaclab.app import AppLauncher

        # Launch Isaac Sim app first
        launcher_args = {
            "headless": headless,
            "enable_cameras": enable_cameras,
            "device": device,
        }
        self._app_launcher = AppLauncher(launcher_args)
        self._app = self._app_launcher.app

        # Import after SimulationApp is initialized
        from isaaclab.envs import ManagerBasedRLEnv

        # Resolve task cfg class
        task_cfg_class = _import_by_path(task_cfg_class_path)
        try:
            import inspect

            print(f"[DexVLAIsaacEnv] task_cfg_class file: {inspect.getfile(task_cfg_class)}")
        except Exception:
            pass
        try:
            scene_cfg = getattr(task_cfg, "scene", None)
            if scene_cfg is not None:
                def _usd_path(asset_cfg):
                    return getattr(getattr(asset_cfg, "spawn", None), "usd_path", None)

                print("[DexVLAIsaacEnv] scene assets:")
                for name in ["table", "peg_block"]:
                    asset_cfg = getattr(scene_cfg, name, None)
                    if asset_cfg is None:
                        continue
                    print(f"  - {name}: {_usd_path(asset_cfg)}")
        except Exception:
            pass
        task_cfg = task_cfg_class()

        # For evaluation, set single-env by default if provided
        if hasattr(task_cfg, "scene") and hasattr(task_cfg.scene, "num_envs"):
            task_cfg.scene.num_envs = 1

        # Create env
        self._env = ManagerBasedRLEnv(task_cfg, render_mode="rgb_array" if enable_render else None)

        # Adapters
        self._obs_adapter = obs_adapter or _default_obs_adapter
        self._action_adapter = action_adapter

        # Optional metadata for video recording
        self.episode_idx = episode_idx
        self.enable_render = enable_render
        self.render_fps = render_fps
        self.crf = crf
        self.output_dir = output_dir
        self.last_video_path: Optional[str] = None

    def reset(self):
        # IsaacLab env reset returns obs dict (batched)
        obs = self._env.reset()
        obs = self._obs_adapter(obs)
        return obs

    def step(self, action: np.ndarray):
        # Convert DP action -> Isaac action if needed
        if self._action_adapter is not None:
            action = self._action_adapter(action)

        # Isaac env expects torch tensor (num_envs, action_dim)
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        action = torch.from_numpy(action).to(self._env.device)

        obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated[0] or truncated[0]) if hasattr(terminated, "__len__") else bool(terminated)

        obs = self._obs_adapter(obs)

        # Reward to scalar
        reward_val = float(reward[0]) if hasattr(reward, "__len__") else float(reward)

        # info: success flag if provided by Isaac env
        success = None
        if isinstance(info, dict):
            # common key used in IsaacLab "extras"
            success = info.get("success", None)
            if success is None and "log" in info and isinstance(info["log"], dict):
                success = info["log"].get("success", None)
            if hasattr(success, "__len__"):
                success = bool(success[0])
        info_out = {"success": success} if success is not None else {}

        return obs, reward_val, done, info_out

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
        if self._app is not None:
            self._app.close()
            self._app = None


def _default_obs_adapter(obs) -> Dict[str, np.ndarray]:
    """
    Default observation adapter:
    - If obs is dict and has "policy" group, returns obs["policy"].
    - Otherwise, returns obs as-is.

    Expected return: dict with "image" (H,W,3) and "qpos" (14,)
    """
    # Some IsaacLab versions return (obs, info)
    if isinstance(obs, (tuple, list)) and len(obs) > 0:
        obs = obs[0]

    if isinstance(obs, dict) and "policy" in obs:
        obs = obs["policy"]

    # If obs already matches desired format, pass through.
    if isinstance(obs, dict) and ("image" in obs or "cam_high" in obs) and (
        "qpos" in obs or ("qpos_left" in obs and "qpos_right" in obs)
    ):
        image = obs["image"] if "image" in obs else obs["cam_high"]
        if "qpos" in obs:
            qpos = obs["qpos"]
        else:
            qpos_left = obs["qpos_left"]
            qpos_right = obs["qpos_right"]
            if hasattr(qpos_left, "cpu"):
                qpos_left = qpos_left.cpu().numpy()
            if hasattr(qpos_right, "cpu"):
                qpos_right = qpos_right.cpu().numpy()
            qpos = np.concatenate([qpos_left, qpos_right], axis=-1)
        if hasattr(image, "cpu"):
            image = image.cpu().numpy()
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu().numpy()
        # If batched, take the first env
        if image.ndim >= 4:
            image = image[0]
        if qpos.ndim >= 2:
            qpos = qpos[0]
        return {"image": image, "qpos": qpos}

    raise ValueError(
        "Unsupported observation format from Isaac env. "
        "Provide obs_adapter to map env observations to {'image', 'qpos'}."
    )


def _import_by_path(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)
