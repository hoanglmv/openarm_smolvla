"""SmolVLA reading one RGB-D image, as ACT does (configuration_smolvla_rgbd).

Two changes to LeRobot's SmolVLAPolicy:

    patch embedding  SigLIP's Conv2d(3 -> 768, 16x16) becomes Conv2d(4 -> 768);
                     loading 3-channel weights (smolvla_base) fills the 4th
                     channel per config.depth_init. Trainable, like ACT's
                     conv1_adapter, with the rest of the encoder frozen.
    prepare_images   the depth camera's first channel is stacked onto the RGB
                     camera as channel 4 before SmolVLA's letterboxing and
                     [-1, 1] scaling, which then apply to all four channels:
                     depth 0..1 (0.2..1.2 m, ACT's scaling) becomes -1..1.
"""

from __future__ import annotations

import logging

import torch
from safetensors.torch import load_file
from torch import nn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from openarm_smolvla.configuration_smolvla_rgbd import SmolVLARGBDConfig

log = logging.getLogger(__name__)
PATCH_WEIGHT = "vision_model.embeddings.patch_embedding.weight"


def widen_patch_weight(weight: torch.Tensor, depth_init: str) -> torch.Tensor:
    """[out, 3, k, k] -> [out, 4, k, k], the 4th channel per depth_init."""
    if depth_init == "mean":
        depth = weight.mean(dim=1, keepdim=True)
    else:
        depth = torch.zeros_like(weight[:, :1])
    return torch.cat([weight, depth], dim=1)


class SmolVLARGBDPolicy(SmolVLAPolicy):
    config_class = SmolVLARGBDConfig
    name = "smolvla_rgbd"

    def __init__(self, config: SmolVLARGBDConfig, **kwargs):
        super().__init__(config, **kwargs)
        vision = self.model.vlm_with_expert.get_vlm_model().vision_model
        old = vision.embeddings.patch_embedding
        trainable = bool(config.train_patch_embedding) or old.weight.requires_grad
        if old.in_channels == 3:
            new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding,
                            bias=old.bias is not None).to(device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                new.weight.copy_(widen_patch_weight(old.weight, config.depth_init))
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            vision.embeddings.patch_embedding = new
        for parameter in vision.embeddings.patch_embedding.parameters():
            parameter.requires_grad = trainable

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        """A 3-channel checkpoint (smolvla_base) loads too: its patch embedding is widened."""
        state = load_file(model_file, device="cpu")
        for key, value in list(state.items()):
            if key.endswith(PATCH_WEIGHT) and value.ndim == 4 and value.shape[1] == 3:
                state[key] = widen_patch_weight(value, model.config.depth_init)
                log.info("%s: 3 channels widened to 4 (depth: %s)", key, model.config.depth_init)
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing or unexpected:
            log.warning("loading %s: %d missing keys %s, %d unexpected %s", model_file,
                        len(missing), missing[:5], len(unexpected), unexpected[:5])
        return model

    def prepare_images(self, batch):
        rgb_key, depth_key = self.config.rgb_key, self.config.depth_key
        if depth_key not in batch:
            raise KeyError(f"no {depth_key} in the batch: this model reads RGB-D")
        batch = dict(batch)
        depth = batch.pop(depth_key)
        channel = depth[:, :, :1] if depth.ndim == 5 else depth[:, :1]
        batch[rgb_key] = torch.cat([batch[rgb_key], channel.to(batch[rgb_key].dtype)], dim=-3)
        return super().prepare_images(batch)
