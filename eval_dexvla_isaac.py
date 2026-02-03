"""
Evaluate a DexVLA Diffusion Policy checkpoint in Isaac Sim (IsaacLab manager-based env).

Run inside env_isaaclab:
python eval_dexvla_isaac.py \
  --checkpoint data/outputs/.../checkpoints/latest.ckpt \
  --task_cfg_class isaaclab_tasks.manager_based.manipulation.deploy.reach.reach_env_cfg.ReachEnvCfg \
  --isaaclab_source /home/hyeseong/IsaacLab/source \
  --device cuda:0 \
  --output_dir data/isaac_eval_dexvla
"""

import os
import pathlib
import click
import dill
import hydra
import torch
import wandb
import h5py
import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.dexvla_isaac_runner import DexVLAIsaacRunner
from diffusion_policy.isaacsim_env.dexvla_isaac_env import DexVLAIsaacEnv
from diffusion_policy.model.language.distilbert_encoder import DistilBertLanguageEncoder


@click.command()
@click.option("-c", "--checkpoint", required=True, type=str)
@click.option("-o", "--output_dir", required=True, type=str)
@click.option("--device", default="cuda:0", type=str)
@click.option("--task_cfg_class", required=True, type=str,
              help="Python path to IsaacLab manager-based env config class.")
@click.option("--isaaclab_source", default=None, type=str,
              help="Path to IsaacLab/source for imports.")
@click.option("--headless/--gui", default=True)
@click.option("--enable_cameras/--disable_cameras", default=True)
@click.option("--n_episodes", default=20, type=int)
@click.option("--max_steps", default=200, type=int)
@click.option("--render_fps", default=10, type=int)
@click.option("--crf", default=22, type=int)
@click.option("--wandb_project", default="diffusion_policy_dexvla", type=str)
@click.option("--wandb_name", default="dexvla_isaac_eval", type=str)
@click.option("--use_language/--no_language", default=True)
@click.option("--language_raw", default=None, type=str)
@click.option("--substep_hdf5", default=None, type=str,
              help="Path to an episode hdf5 to infer substep_reasonings change points.")
@click.option("--substep_dir", default=None, type=str,
              help="Directory containing episode_*.hdf5 files for per-episode substeps.")
@click.option("--substep_pattern", default="episode_*.hdf5", type=str,
              help="Glob pattern for episode files inside --substep_dir.")
def main(checkpoint, output_dir, device, task_cfg_class, isaaclab_source,
         headless, enable_cameras, n_episodes, max_steps, render_fps, crf,
         wandb_project, wandb_name, use_language, language_raw,
         substep_hdf5, substep_dir, substep_pattern):
    output_dir = str(pathlib.Path(output_dir))
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint (same flow as eval.py)
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema and getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model
    policy.to(torch.device(device))
    policy.eval()

    # language provider (Cook rice + substep_reasonings)
    language_provider = None
    if use_language:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        encoder = DistilBertLanguageEncoder(device=torch.device(device))

        # infer substep segments from hdf5 (optional)
        episode_files = []
        if substep_dir is not None:
            episode_files = _collect_episode_files(substep_dir, substep_pattern)

        # cache for per-episode selection
        per_episode_cache = {}

        def _get_episode_substeps(ep_idx: int):
            if ep_idx in per_episode_cache:
                return per_episode_cache[ep_idx]

            if len(episode_files) > 0:
                # choose episode file by index (wrap-around)
                ep_path = episode_files[ep_idx % len(episode_files)]
            elif substep_hdf5 is not None:
                ep_path = os.path.expanduser(substep_hdf5)
            else:
                ep_path = None

            if ep_path is not None:
                ep_language_raw = language_raw or _read_language_raw(ep_path) or "Cook rice."
                ep_segments = _read_substep_segments(ep_path)
            else:
                ep_language_raw = language_raw or "Cook rice."
                ep_segments = []

            per_episode_cache[ep_idx] = (ep_language_raw, ep_segments)
            return per_episode_cache[ep_idx]

        def language_provider(ep_idx: int, step_idx: int, to_steps: int) -> torch.Tensor:
            # choose substep based on current step index (per-episode)
            ep_language_raw, ep_segments = _get_episode_substeps(ep_idx)
            text = ep_language_raw
            for seg in ep_segments:
                if seg["start"] <= step_idx <= seg["end"]:
                    text = seg["text"]
                    break
            emb = encoder.encode([text])  # (1, 768)
            # expand to (B=1, To, 768)
            emb = emb.unsqueeze(1).repeat(1, to_steps, 1)
            return emb

    # wandb init
    wandb_run = wandb.init(
        project=wandb_project,
        name=wandb_name,
        dir=output_dir,
        config={
            "checkpoint": checkpoint,
            "task_cfg_class": task_cfg_class,
            "device": device,
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "use_language": use_language,
            "language_raw": language_raw,
            "substep_hdf5": substep_hdf5,
            "substep_dir": substep_dir,
            "substep_pattern": substep_pattern,
        },
    )

    # Runner with Isaac env
    runner = DexVLAIsaacRunner(
        output_dir=output_dir,
        env_cls=DexVLAIsaacEnv,
        n_episodes=n_episodes,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_steps=max_steps,
        render_fps=render_fps,
        crf=crf,
        reuse_env=True,
        language_provider=language_provider,
        env_kwargs={
            "task_cfg_class_path": task_cfg_class,
            "isaaclab_source": isaaclab_source,
            "headless": headless,
            "enable_cameras": enable_cameras,
            "device": device,
        },
    )

    log_data = runner.run(policy)
    wandb_run.log(log_data)
    wandb_run.finish()

    # Save metrics locally
    metrics_path = os.path.join(output_dir, "eval_log.json")
    with open(metrics_path, "w") as f:
        import json
        json.dump(log_data, f, indent=2, sort_keys=True, default=str)

    print("test/mean_score:", log_data.get("test/mean_score"))
    print("metrics saved to:", metrics_path)


def _read_language_raw(hdf5_path: str) -> str | None:
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "language_raw" not in f:
                return None
            raw = f["language_raw"][:]
            if raw.size == 0:
                return None
            value = raw[0]
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if isinstance(value, np.bytes_):
                return value.decode("utf-8")
            return str(value)
    except Exception:
        return None


def _read_substep_segments(hdf5_path: str) -> list[dict]:
    """
    Parse substep_reasonings array into contiguous segments:
    [{"start": i, "end": j, "text": "..."}, ...]
    """
    segments = []
    with h5py.File(hdf5_path, "r") as f:
        if "substep_reasonings" not in f:
            return segments
        arr = f["substep_reasonings"][:]
        if arr.size == 0:
            return segments

        # Decode to strings if possible
        texts = []
        for v in arr:
            if isinstance(v, (bytes, np.bytes_)):
                texts.append(v.decode("utf-8"))
            else:
                texts.append(str(v))

        # Build contiguous segments
        start = 0
        current = texts[0]
        for idx in range(1, len(texts)):
            if texts[idx] != current:
                segments.append({"start": start, "end": idx - 1, "text": current})
                start = idx
                current = texts[idx]
        segments.append({"start": start, "end": len(texts) - 1, "text": current})
    return segments


def _collect_episode_files(substep_dir: str, pattern: str) -> list[str]:
    import glob
    substep_dir = os.path.expanduser(substep_dir)
    files = sorted(glob.glob(os.path.join(substep_dir, pattern)))
    return files


if __name__ == "__main__":
    main()
