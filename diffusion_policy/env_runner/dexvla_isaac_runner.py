import os
import math
import pathlib
import collections
from typing import Dict, Optional, Type, Callable

import imageio

import numpy as np
import torch
import tqdm
import wandb
import wandb.sdk.data_types.video as wv

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply


class DexVLAIsaacRunner(BaseImageRunner):
    """
    Isaac Sim 기반 DexVLA 태스크 평가용 runner.

    기존 `PushTImageRunner` / `RobomimicImageRunner` 와 동일한 인터페이스를 따르며,
    - 여러 에피소드 rollout
    - wandb에 success rate(`test/mean_score`)와 에피소드 영상(`test/sim_video_*`)을 로깅할 수 있는
      metrics dict를 반환합니다.

    NOTE:
        실제 Isaac Sim 환경 래퍼 클래스(DexVLAIsaacEnv)는 별도 패키지에서 제공된다고 가정하고,
        `env_cls` 인자로 주입받는 형태로 구현했습니다.
        예: from your_isaac_pkg.dexvla_env import DexVLAIsaacEnv
    """

    def __init__(
        self,
        output_dir: str,
        # Isaac env 클래스 (Base Isaac Env 타입)
        env_cls: Type,
        n_episodes: int = 20,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        max_steps: int = 200,
        render_fps: int = 10,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        reuse_env: bool = True,
        env_kwargs: Optional[dict] = None,
        language_provider: Optional[Callable[[int, int, int], torch.Tensor]] = None,
    ):
        super().__init__(output_dir)

        self.env_cls = env_cls
        self.n_episodes = n_episodes
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.render_fps = render_fps
        self.crf = crf
        self.tqdm_interval_sec = tqdm_interval_sec
        self.reuse_env = reuse_env
        self.env_kwargs = env_kwargs or {}
        self._env = None
        self.language_provider = language_provider

    def _create_env(self, episode_idx: int, enable_render: bool = True):
        """
        Isaac Sim 환경 인스턴스를 생성하는 helper.

        DexVLAIsaacEnv 의 __init__ 시그니처에 맞게 필요한 인자를 넘겨주세요.
        """
        env = self.env_cls(
            episode_idx=episode_idx,
            enable_render=enable_render,
            render_fps=self.render_fps,
            crf=self.crf,
            output_dir=self.output_dir,
            **self.env_kwargs,
        )
        return env

    def run(self, policy: BaseImagePolicy) -> Dict:
        device = policy.device

        all_rewards = []
        all_video_paths = []
        all_success = []

        for ep in range(self.n_episodes):
            enable_render = True  # 필요하면 ep < n_vis 로 바꿀 수 있음
            if self.reuse_env:
                if self._env is None:
                    self._env = self._create_env(ep, enable_render=enable_render)
                env = self._env
            else:
                env = self._create_env(ep, enable_render=enable_render)

            obs = env.reset()
            policy.reset()

            ep_rewards = []
            done = False
            ep_success: Optional[bool] = None

            # progress bar
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval DexVLAIsaac {ep+1}/{self.n_episodes}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            # observation history buffers (To = n_obs_steps)
            obs_history_image = collections.deque(maxlen=self.n_obs_steps)
            obs_history_qpos = collections.deque(maxlen=self.n_obs_steps)

            # initialize history with the first obs (pad by repeating)
            obs_history_image.clear()
            obs_history_qpos.clear()
            for _ in range(self.n_obs_steps):
                obs_history_image.append(obs["image"])
                obs_history_qpos.append(obs["qpos"])

            # video frames (from cam_high)
            frames = []
            # include first obs frame
            if enable_render and isinstance(obs, dict) and "image" in obs:
                frames.append(obs["image"])

            step_idx = 0
            while not done:
                # Isaac env 에서 받은 obs -> DP 포맷으로 변환
                # DexVLAImageLanguageDataset 와 동일한 키/shape 를 맞춰야 함.
                image = obs["image"]  # (H, W, 3), uint8 or float
                qpos = obs["qpos"]    # (14,)

                # update history
                obs_history_image.append(image)
                obs_history_qpos.append(qpos)

                image_t = image.astype(np.float32) / 255.0
                image_t = np.moveaxis(image_t, -1, 0)  # (3, H, W)

                To = self.n_obs_steps
                # stack history (To, H, W, 3) -> (To, 3, H, W)
                image_hist = np.stack([img.astype(np.float32) / 255.0 for img in obs_history_image], axis=0)
                image_hist = np.moveaxis(image_hist, -1, 1)  # (To, 3, H, W)
                qpos_hist = np.stack([qp.astype(np.float32) for qp in obs_history_qpos], axis=0)  # (To, 14)

                image_bt = image_hist[None, ...]  # (1, To, 3, H, W)
                qpos_bt = qpos_hist[None, ...]    # (1, To, 14)
                obs_dict_np = {
                    "image": image_bt,
                    "qpos": qpos_bt,
                }

                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: torch.from_numpy(x).to(device=device),
                )

                # 언어 임베딩 (optional): step index 기반 provider 사용
                language_emb = None
                if self.language_provider is not None:
                    language_emb = self.language_provider(ep, step_idx, self.n_obs_steps)

                with torch.no_grad():
                    out = policy.predict_action(obs_dict, language_emb=language_emb)

                # 한 step action만 사용 (Ta >= 1인 경우 첫 스텝)
                action = out["action"][0, 0].detach().cpu().numpy()

                obs, reward, done, info = env.step(action)
                done = bool(done)
                ep_rewards.append(float(reward))
                if isinstance(info, dict) and ("success" in info):
                    ep_success = bool(info["success"])

                if enable_render and isinstance(obs, dict) and "image" in obs:
                    frames.append(obs["image"])

                pbar.update(1)
                step_idx += 1

                if len(ep_rewards) >= self.max_steps:
                    break

            pbar.close()

            # 에피소드별 reward 기록
            all_rewards.append(ep_rewards)

            # 비디오 저장 (cam_high 프레임 기반)
            video_path = None
            if enable_render and len(frames) > 0:
                media_dir = pathlib.Path(self.output_dir).joinpath("media")
                media_dir.mkdir(parents=True, exist_ok=True)
                filename = media_dir.joinpath(f"episode_{ep:04d}.mp4")
                imageio.mimsave(str(filename), frames, fps=self.render_fps)
                video_path = str(filename)

            # 에피소드 단위로 wandb 즉시 로깅
            if wandb.run is not None:
                seed = ep
                prefix = "test/"
                ep_max_r = max(ep_rewards) if len(ep_rewards) > 0 else 0.0
                ep_log = {
                    prefix + f"sim_max_reward_{seed}": ep_max_r,
                }
                if video_path is not None and os.path.isfile(video_path):
                    ep_log[prefix + f"sim_video_{seed}"] = wandb.Video(video_path)
                if ep_success is not None:
                    ep_log[prefix + f"success_{seed}"] = int(ep_success)
                wandb.log(ep_log)
            all_video_paths.append(video_path)
            all_success.append(ep_success)

            if not self.reuse_env:
                env.close()

        # wandb 로깅용 metrics dict 구성
        log_data: Dict[str, object] = dict()
        episode_max_rewards = [max(rs) if len(rs) > 0 else 0.0 for rs in all_rewards]

        for idx, (max_r, video_path) in enumerate(zip(episode_max_rewards, all_video_paths)):
            seed = idx  # 에피소드 인덱스를 pseudo-seed 로 사용
            prefix = "test/"
            log_data[prefix + f"sim_max_reward_{seed}"] = max_r

            if video_path is not None and os.path.isfile(video_path):
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}"] = sim_video

        # success rate (mean_score) 계산: max reward 기준 평균
        # If success flags are available, use them; otherwise fall back to max reward.
        valid_success = [s for s in all_success if s is not None]
        if len(valid_success) > 0:
            mean_score = float(np.mean(valid_success))
        else:
            mean_score = float(np.mean(episode_max_rewards)) if len(episode_max_rewards) > 0 else 0.0
        log_data["test/mean_score"] = mean_score

        if self.reuse_env and self._env is not None:
            self._env.close()
            self._env = None

        return log_data

