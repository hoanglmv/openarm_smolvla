#!/usr/bin/env python3
"""Train from the Hub dataset in one command, every step only when needed:

    1. wandb      the key is there (before hours of downloading, not after)
    2. download   the episodes not under data/raw/<repo name>/ yet
    3. convert    unless data/lerobot/<repo_id> holds exactly these episodes,
                  tasks and settings (a stale or partial one is replaced)
    4. norm stats recomputed (seconds)
    5. train

    uv run scripts/train_pipeline.py exp_name=v1
    uv run scripts/train_pipeline.py exp_name=v1 batch_size=32 finetune=full
    uv run scripts/train_pipeline.py exp_name=v1 resume=true        # continue a run
    uv run scripts/train_pipeline.py exp_name=try download.limit=10 num_train_steps=2000
    uv run scripts/train_pipeline.py exp_name=v1 download.enable=false   # episodes already in data/raw
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from openarm_smolvla import paths


def step(title: str) -> None:
    print(f"\n=== {title}", flush=True)


@hydra.main(version_base="1.3", config_path="../conf", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    logging.getLogger().setLevel(logging.INFO)
    paths.setup_env(cfg)
    from openarm_smolvla.convert import convert
    from openarm_smolvla.download import check
    from openarm_smolvla.download import destination
    from openarm_smolvla.download import download
    from openarm_smolvla.episode import find_episodes
    from openarm_smolvla.norm_stats import compute_norm_stats
    from openarm_smolvla.run import check_wandb
    from openarm_smolvla.run import train

    step("1/5 wandb")
    check_wandb(cfg)
    print("off" if not cfg.wandb.enable else f"{cfg.wandb.mode}, project {cfg.wandb.project}")

    step("2/5 episodes")
    download_cfg = OmegaConf.merge({"paths": cfg.paths}, cfg.download)
    folder = destination(download_cfg)
    print(f"{len(find_episodes(folder)) if folder.is_dir() else 0} episodes in {folder}")
    if cfg.download.enable:
        result = download(download_cfg)
        print(f"{result['repo_id']}@{result['sha'][:8]}: {len(result['files'])} episodes, "
              f"{len(result['fetched'])} fetched now")
        if cfg.download.check and result["fetched"]:
            problems = check(Path(result["dir"]), result["fetched"])
            for name, found in problems.items():
                print(f"  {name}: {'; '.join(found)}")
    # Every episode under paths.raw is converted, as with convert_dataset.py;
    # the manifest excludes or holds out the ones not to train on.

    step("3/5 LeRobot dataset")
    convert_cfg = OmegaConf.merge(
        {"paths": cfg.paths, "data": cfg.data, "overwrite": False, "reuse_current": True, "dry_run": False},
        cfg.convert,
    )
    report = convert(convert_cfg)
    print(f"{report['repo_id']}: {len(report['episodes'])} episodes, {report['frames']} frames, "
          f"{len(report['skipped'])} skipped" + (" (already converted)" if report.get("reused") else ""))

    step("4/5 norm stats")
    print(compute_norm_stats(cfg))

    step("5/5 train")
    print(f"run finished: {train(cfg)}")


if __name__ == "__main__":
    main()
