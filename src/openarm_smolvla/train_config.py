"""Hydra config -> LeRobot's SmolVLAConfig and TrainPipelineConfig.

The Hydra tree (conf/) holds every choice; this is the one place that turns
it into LeRobot's dataclasses, for training, serving and evaluation alike.
LeRobot's command line (draccus) is not used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf

from openarm_smolvla import constants as C
from openarm_smolvla import paths


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def policy_overrides(cfg: DictConfig) -> dict:
    """The SmolVLAConfig fields this repo sets, from model + finetune."""
    model, finetune = cfg.model, cfg.finetune
    overrides = {
        "chunk_size": int(model.chunk_size),
        "n_action_steps": int(model.n_action_steps),
        "num_steps": int(model.num_steps),
        "resize_imgs_with_padding": tuple(int(v) for v in model.resize_imgs_with_padding),
        "empty_cameras": int(model.empty_cameras),
        "tokenizer_max_length": int(model.tokenizer_max_length),
        "use_amp": bool(model.use_amp),
        "freeze_vision_encoder": bool(finetune.freeze_vision_encoder),
        "train_expert_only": bool(finetune.train_expert_only),
        "train_state_proj": bool(finetune.train_state_proj),
        # The SmolVLA checkpoint holds every weight, the VLM's included: build
        # SmolVLM2 from its config and skip downloading its own weights.
        "load_vlm_weights": False,
        "device": resolve_device(str(cfg.device)),
        "push_to_hub": False,
        "repo_id": None,
        # Filled from the dataset by make_policy: chest (+ chest_depth), 16 joints.
        "input_features": {},
        "output_features": {},
    }
    if finetune.get("num_vlm_layers"):
        overrides["num_vlm_layers"] = int(finetune.num_vlm_layers)
    return overrides


def rgbd_fields(cfg: DictConfig) -> dict | None:
    """With data.depth_as=channel, ACT's input: depth as the RGB image's 4th
    channel (configuration_smolvla_rgbd); None for SmolVLA's own input."""
    depth_as = str(cfg.data.get("depth_as") or "image")
    if depth_as == "image":
        return None
    if depth_as != "channel":
        raise ValueError(f"data.depth_as must be image or channel, not {depth_as!r}")
    cameras = [str(c) for c in cfg.data.cameras]
    if cameras != [C.CAMERA_RGB, C.CAMERA_DEPTH]:
        raise ValueError(f"data.depth_as=channel needs cameras [chest, chest_depth], got {cameras}")
    return {
        "rgb_key": C.LEROBOT_IMAGE_PREFIX + C.CAMERA_RGB,
        "depth_key": C.LEROBOT_IMAGE_PREFIX + C.CAMERA_DEPTH,
        "depth_init": str(cfg.data.get("depth_init") or "zero"),
        "train_patch_embedding": bool(cfg.finetune.get("train_patch_embedding", True)),
    }


def build_policy_config(cfg: DictConfig):
    """-> SmolVLAConfig (or SmolVLARGBDConfig with data.depth_as=channel): the
    base model's own config with this repo's overrides, or a fresh one
    (finetune.load_base false)."""
    import dataclasses

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    from openarm_smolvla.configuration_smolvla_rgbd import SmolVLARGBDConfig

    overrides = policy_overrides(cfg)
    rgbd = rgbd_fields(cfg)
    kind = SmolVLAConfig if rgbd is None else SmolVLARGBDConfig
    overrides.update(rgbd or {})
    if not cfg.finetune.load_base:
        return kind(**overrides)
    base = str(cfg.model.base)
    revision = cfg.model.get("base_revision")
    config = PreTrainedConfig.from_pretrained(base, revision=revision)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError(f"{base} is a {config.type} checkpoint, not SmolVLA")
    fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(SmolVLAConfig) if f.init}
    config = kind(**(fields | overrides))
    config.pretrained_path = base
    config.pretrained_revision = revision
    return config


def build_train_config(cfg: DictConfig, output_dir: Path):
    """-> lerobot.configs.train.TrainPipelineConfig for the composed config."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.default import WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.optim import AdamWConfig
    from lerobot.optim import CosineDecayWithWarmupSchedulerConfig
    from lerobot.transforms import ImageTransformsConfig

    from openarm_smolvla.video import video_backend

    if cfg.lr.warmup_steps >= cfg.lr.decay_steps:
        raise ValueError(f"lr.warmup_steps ({cfg.lr.warmup_steps}) must be below lr.decay_steps ({cfg.lr.decay_steps})")
    if cfg.model.n_action_steps > cfg.model.chunk_size:
        raise ValueError(f"model.n_action_steps ({cfg.model.n_action_steps}) exceeds model.chunk_size")

    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=str(cfg.data.repo_id),
            image_transforms=ImageTransformsConfig(enable=bool(cfg.data.image_transforms)),
            video_backend=video_backend(),
        ),
        policy=build_policy_config(cfg),
        output_dir=Path(output_dir),
        job_name=f"{cfg.name}_{cfg.exp_name}",  # the wandb run's name
        seed=int(cfg.seed),
        num_workers=int(cfg.num_workers),
        batch_size=int(cfg.batch_size),
        steps=int(cfg.num_train_steps),
        log_freq=int(cfg.log_interval),
        save_freq=int(cfg.save_interval),
        env_eval_freq=0,
        use_policy_training_preset=False,
        optimizer=AdamWConfig(
            lr=float(cfg.lr.peak_lr),
            betas=tuple(float(b) for b in cfg.optimizer.betas),
            eps=float(cfg.optimizer.eps),
            weight_decay=float(cfg.optimizer.weight_decay),
            grad_clip_norm=float(cfg.optimizer.grad_clip_norm),
        ),
        scheduler=CosineDecayWithWarmupSchedulerConfig(
            peak_lr=float(cfg.lr.peak_lr),
            decay_lr=float(cfg.lr.decay_lr),
            num_warmup_steps=int(cfg.lr.warmup_steps),
            num_decay_steps=int(cfg.lr.decay_steps),
        ),
        wandb=WandBConfig(
            enable=bool(cfg.wandb.enable),
            project=str(cfg.wandb.project),
            entity=cfg.wandb.entity,
            mode=str(cfg.wandb.mode),
            notes=cfg.wandb.notes,
            disable_artifact=not bool(cfg.wandb.upload_checkpoints),
        ),
    )


def policy_metadata(cfg: DictConfig) -> dict:
    """What the server tells a client when it connects (openarm_smolvla_client
    reads it; the keys are the ones openarm_pizero's server sends, too)."""
    return {
        "robot": "openarm_v1_bimanual",
        "joint_names": list(C.JOINT_NAMES),
        "action_dim": C.NUM_JOINTS,
        "action_horizon": int(cfg.model.chunk_size),
        "fps": int(cfg.data.fps),
        "model": str(cfg.model.name),
        "finetune": str(cfg.finetune.name),
        "cameras": [str(c) for c in cfg.data.cameras],
        "delta_actions": bool(cfg.data.delta_actions),
        "depth_as": str(cfg.data.get("depth_as") or "image"),  # channel: one RGB-D image, as ACT
        "default_prompt": default_prompt(cfg),
        # openarm_act refuses a camera that gives other frames than these.
        "image_size": dataset_image_size(cfg),
    }


def default_prompt(cfg: DictConfig) -> str | None:
    """data.default_prompt, else the dataset's task when it has only one."""
    if cfg.data.get("default_prompt"):
        return str(cfg.data.default_prompt)
    tasks = cfg.data.get("tasks") or dataset_tasks(cfg)
    return str(tasks[0]) if tasks and len(tasks) == 1 else None


def _dataset_meta_dir(cfg: DictConfig) -> Path:
    home = os.environ.get("HF_LEROBOT_HOME") or str(paths.resolve(cfg.paths.lerobot_home))
    return Path(home) / str(cfg.data.repo_id) / "meta"


def dataset_tasks(cfg: DictConfig) -> list[str] | None:
    """The tasks (prompts) of the converted dataset, from meta/tasks.parquet."""
    tasks = _dataset_meta_dir(cfg) / "tasks.parquet"
    if not tasks.exists():
        return None
    import pyarrow.parquet as pq

    table = pq.read_table(tasks)
    names = table.column("task").to_pylist() if "task" in table.column_names else None
    if names is None:  # the task strings are the index
        names = list(table.to_pandas().index)
    return [str(n) for n in names]


def dataset_image_size(cfg: DictConfig) -> list[int] | None:
    """[height, width] of the dataset's frames: data.image_size, else what the
    converter wrote into the LeRobot dataset's meta/info.json; None if neither."""
    if cfg.data.get("image_size"):
        return [int(v) for v in cfg.data.image_size]
    info = _dataset_meta_dir(cfg) / "info.json"
    if not info.exists():
        return None
    camera = str(cfg.data.cameras[0])
    shape = json.loads(info.read_text())["features"][C.LEROBOT_IMAGE_PREFIX + camera]["shape"]
    return [int(shape[0]), int(shape[1])]


def save_run_config(cfg: DictConfig, directory: Path) -> Path:
    path = Path(directory) / paths.RUN_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)
    return path
