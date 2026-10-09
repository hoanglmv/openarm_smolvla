#!/usr/bin/env python3
"""A trained checkpoint -> a release: weights, processors, config, tokenizer, metadata.

    uv run scripts/export_policy.py checkpoint=checkpoints/smolvla_expert/v1/checkpoints/020000
    uv run scripts/export_policy.py checkpoint=checkpoints/smolvla_expert/v1 repo_id=<user>/openarm-smolvla
    uv run scripts/serve.py checkpoint=releases/smolvla_expert_v1_020000
"""

import logging

import hydra
from omegaconf import DictConfig

from openarm_smolvla import paths
from openarm_smolvla.export import export


@hydra.main(version_base="1.3", config_path="../conf", config_name="export")
def main(cfg: DictConfig) -> None:
    paths.load_dotenv()
    logging.getLogger().setLevel(logging.INFO)
    print(f"release written to {export(cfg)}")


if __name__ == "__main__":
    main()
