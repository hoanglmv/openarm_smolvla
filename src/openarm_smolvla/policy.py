"""A trained SmolVLA, taking OpenArm observations as openarm_act hands them over.

    infer({"images": {"chest": uint8 [H, W, 3], "chest_depth": uint16 [H, W] mm},
           "state": float32 [16] qpos,
           "prompt": str (optional)})
        -> {"actions": float32 [chunk, 16]}  absolute targets, the dataset's
                                             joint order and units

The observation is what openarm_smolvla_client sends; the server and the
offline evaluation both call this. The images go through images.py, as the
converter's did, and then the processors saved with the checkpoint:
tokenising the prompt, normalising with the run's stats, delta actions when
the run used them. Nothing about the pipeline is rebuilt from defaults.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from openarm_smolvla import constants as C
from openarm_smolvla.images import to_model_input
from openarm_smolvla.run import checkpoint_step
from openarm_smolvla.run import load_run_config
from openarm_smolvla.train_config import default_prompt
from openarm_smolvla.train_config import policy_metadata
from openarm_smolvla.train_config import resolve_device

log = logging.getLogger(__name__)
# Where export.py puts SmolVLM2's config, tokenizer and processor (no weights),
# so a release serves without reaching the Hub.
VLM_DIR = "vlm"


class OpenArmPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        default_prompt_override: str | None = None,
        num_steps: int | None = None,
    ):
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.factory import get_policy_class

        import openarm_smolvla.configuration_smolvla_rgbd  # noqa: F401  -- registers "smolvla_rgbd"
        from openarm_smolvla.processors import link_delta_steps  # also registers the steps for loading

        self.run_cfg, self.directory = load_run_config(checkpoint)
        config = PreTrainedConfig.from_pretrained(self.directory)
        config.device = resolve_device(device)
        config.pretrained_path = self.directory
        if num_steps:
            config.num_steps = int(num_steps)
        preprocessor_overrides = {"device_processor": {"device": config.device}}
        vlm = self.directory / VLM_DIR
        if vlm.is_dir():
            config.vlm_model_name = str(vlm)
            preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": str(vlm)}

        # smolvla, or smolvla_rgbd (ACT's 4-channel RGB-D input): config.json says which.
        self.policy = get_policy_class(config.type).from_pretrained(self.directory, config=config)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config, pretrained_path=str(self.directory), preprocessor_overrides=preprocessor_overrides
        )
        link_delta_steps(self.preprocessor, self.postprocessor)
        self.device = config.device
        self.cameras = [str(c) for c in self.run_cfg.data.cameras]
        self.chunk_size = int(config.chunk_size)
        self.default_prompt = default_prompt_override or default_prompt(self.run_cfg)
        self.metadata = policy_metadata(self.run_cfg)
        self.metadata["default_prompt"] = self.default_prompt
        self.metadata["checkpoint"] = str(self.directory)
        self.metadata["step"] = checkpoint_step(self.directory)
        if self.metadata["step"] is None and (self.directory / "metadata.json").exists():  # a release
            self.metadata["step"] = json.loads((self.directory / "metadata.json").read_text()).get("step")
        self.metadata["num_steps"] = int(config.num_steps)
        log.info("loaded %s on %s: cameras %s, chunk %d", self.directory, self.device, self.cameras, self.chunk_size)

    def batch(self, obs: dict) -> dict:
        """The observation as LeRobot's dataset yields a frame (no batch dim)."""
        import torch

        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape != (C.NUM_JOINTS,):
            raise ValueError(f"state has {state.shape[0]} values, expected {C.NUM_JOINTS}")
        images = obs.get("images") or {}
        missing = [c for c in self.cameras if c not in images]
        if missing:
            raise KeyError(f"no image from camera(s) {missing}; got {sorted(images)}")
        prompt = obs.get("prompt") or self.default_prompt
        if isinstance(prompt, bytes):
            prompt = prompt.decode()
        if not prompt:
            raise ValueError("no prompt: send one, or serve with default_prompt=...")
        batch = {C.LEROBOT_STATE: torch.from_numpy(state), C.LEROBOT_TASK: str(prompt)}
        for camera in self.cameras:
            batch[C.LEROBOT_IMAGE_PREFIX + camera] = torch.from_numpy(to_model_input(images[camera]))
        return batch

    def infer(self, obs: dict) -> dict:
        import torch

        with torch.inference_mode():
            batch = self.preprocessor(self.batch(obs))
            actions = self.policy.predict_action_chunk(batch)
            actions = self.postprocessor(actions)
        actions = actions.to(torch.float32).cpu().numpy()[0]
        if actions.shape != (self.chunk_size, C.NUM_JOINTS):
            raise AssertionError(f"the policy returned actions of shape {actions.shape}")
        return {"actions": actions}

    def reset(self) -> None:
        self.policy.reset()
