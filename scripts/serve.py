#!/usr/bin/env python3
"""Serve a checkpoint over websocket for openarm_act (SmolVLAPolicy).

    uv run scripts/serve.py checkpoint=releases/smolvla_expert_first_020000
    uv run scripts/serve.py checkpoint=checkpoints/smolvla_expert/first     # its newest step
"""

import logging

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../conf", config_name="serve")
def main(cfg: DictConfig) -> None:
    logging.getLogger().setLevel(logging.INFO)
    from openarm_smolvla.policy import OpenArmPolicy
    from openarm_smolvla.server import PolicyServer

    policy = OpenArmPolicy(cfg.checkpoint, device=str(cfg.device), default_prompt_override=cfg.default_prompt,
                           num_steps=cfg.num_steps)
    logging.info("serving %s (%s/%s, prompt %r) on port %d", policy.directory, policy.run_cfg.model.name,
                 policy.run_cfg.finetune.name, policy.default_prompt, cfg.port)
    PolicyServer(policy, port=int(cfg.port), metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
