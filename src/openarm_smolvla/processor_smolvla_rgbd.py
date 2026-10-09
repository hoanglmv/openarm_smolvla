"""Processors for policy type "smolvla_rgbd": SmolVLA's own, unchanged.

The depth image travels as its own camera through the processors and is
fused into the RGB one inside the model (modeling_smolvla_rgbd).
"""

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def make_smolvla_rgbd_pre_post_processors(config, dataset_stats=None):
    return make_smolvla_pre_post_processors(config, dataset_stats)
