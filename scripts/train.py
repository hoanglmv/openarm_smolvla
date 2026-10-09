#!/usr/bin/env python3
"""Fine-tune SmolVLA on an OpenArm dataset, with LeRobot's training loop.

    uv run scripts/train.py exp_name=first                      # finetune=expert
    uv run scripts/train.py finetune=full exp_name=first batch_size=32
    uv run scripts/train.py exp_name=first resume=true          # continue
    uv run scripts/train.py experiment=smoke                    # CPU pipeline check

Checkpoints: checkpoints/<model>_<finetune>/<exp_name>/checkpoints/<step>/,
each with the resolved config (openarm_config.yaml) in pretrained_model/.
"""

import hydra
from omegaconf import DictConfig

from openarm_smolvla import paths


@hydra.main(version_base="1.3", config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    paths.setup_env(cfg)
    from openarm_smolvla.run import train

    print(f"run finished: {train(cfg)}")


if __name__ == "__main__":
    main()
