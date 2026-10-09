"""SmolVLA with ACT's RGB-D input: one 4-channel image, depth as its 4th channel.

act_pipeline (OpenArm_MC) feeds ACT one [4, H, W] chest image, RGB and depth
stacked, through a ResNet whose first conv was widened to four channels
(`conv1_adapter`). This does the same to SmolVLA's vision encoder: SigLIP's
patch embedding takes four channels, the rest of SmolVLA is unchanged.

Registered with LeRobot as policy type "smolvla_rgbd". LeRobot finds the
policy (modeling_smolvla_rgbd) and processors (processor_smolvla_rgbd) by
this module's name, and loads a checkpoint of this type once it is imported.
"""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

DEPTH_INITS = ("zero", "mean")


@PreTrainedConfig.register_subclass("smolvla_rgbd")
@dataclass
class SmolVLARGBDConfig(SmolVLAConfig):
    # The RGB camera and the depth camera fused into it. The depth image is the
    # dataset's gray encoding (images.depth_to_image); its first channel is used.
    rgb_key: str = "observation.images.chest"
    depth_key: str = "observation.images.chest_depth"
    # The 4th channel's patch-embedding weights when starting from 3-channel
    # weights. "zero": the image features start exactly as pretrained and
    # depth is learned in; "mean": the mean of the RGB weights, as act_pipeline
    # initialises its conv1_adapter.
    depth_init: str = "zero"
    # Train the patch embedding (as act_pipeline trains conv1_adapter) even
    # when the vision encoder is frozen; it is how depth gets learned.
    train_patch_embedding: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.depth_init not in DEPTH_INITS:
            raise ValueError(f"depth_init must be one of {DEPTH_INITS}, not {self.depth_init!r}")
