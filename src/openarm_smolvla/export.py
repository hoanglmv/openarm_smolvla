"""A trained checkpoint -> a self-contained release for inference.

Training checkpoints (checkpoints/<run>/<exp>/checkpoints/<step>/) carry the
optimizer state, for resuming. A release keeps what serving needs, and
serves without the Hub:

    releases/<name>/
      model.safetensors     the weights (float32, or bfloat16 with dtype=bfloat16)
      config.json           SmolVLA's config
      policy_preprocessor.json / policy_postprocessor.json + their
      *.safetensors         the processors, with the run's norm stats
      openarm_config.yaml   the run's resolved config
      vlm/                  SmolVLM2's config, tokenizer and processor (no weights)
      metadata.json         model, dataset, image size, prompt, source checkpoint

serve.py and offline_eval.py take a release directory like a checkpoint.
With `repo_id`, the release is also uploaded to a Hugging Face model repo.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from omegaconf import DictConfig

from openarm_smolvla import paths
from openarm_smolvla.policy import VLM_DIR
from openarm_smolvla.run import MODEL_FILE
from openarm_smolvla.run import PRETRAINED_DIR
from openarm_smolvla.run import checkpoint_step
from openarm_smolvla.run import load_run_config
from openarm_smolvla.train_config import policy_metadata

log = logging.getLogger(__name__)
# What a pretrained_model/ holds besides the weights that inference reads.
KEEP = ("config.json", "policy_preprocessor*", "policy_postprocessor*", paths.RUN_CONFIG)
# SmolVLM2's files the model and tokenizer are built from; never its weights.
VLM_PATTERNS = ["*.json", "*.txt"]


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(paths.REPO_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_weights(source: Path, target: Path, dtype: str) -> None:
    if dtype == "float32":
        shutil.copy2(source, target)
        return
    if dtype != "bfloat16":
        raise ValueError(f"dtype must be float32 or bfloat16, not {dtype!r}")
    import torch
    from safetensors.torch import load_file
    from safetensors.torch import save_file

    tensors = load_file(source)
    save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in tensors.items()}, target)


def bundle_vlm(source: Path, out: Path) -> None:
    """SmolVLM2's config and tokenizer into out/vlm, from the checkpoint or the Hub."""
    if (source / VLM_DIR).is_dir():
        shutil.copytree(source / VLM_DIR, out / VLM_DIR)
        return
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download

    name = json.loads((source / "config.json").read_text())["vlm_model_name"]
    snapshot_download(name, allow_patterns=VLM_PATTERNS, local_dir=str(out / VLM_DIR))


def export(cfg: DictConfig) -> Path:
    run_cfg, source = load_run_config(cfg.checkpoint)
    step = checkpoint_step(source)
    label = source.parent.name if source.name == PRETRAINED_DIR else source.name  # 020000
    name = cfg.name or f"{run_cfg.name}_{run_cfg.exp_name}_{label}"
    out = paths.resolve(cfg.paths.releases) / name
    if out.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{out} exists; pass overwrite=true to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for pattern in KEEP:
        for file in source.glob(pattern):
            shutil.copy2(file, out / file.name)
    write_weights(source / MODEL_FILE, out / MODEL_FILE, str(cfg.dtype))
    bundle_vlm(source, out)

    metadata = {
        **policy_metadata(run_cfg),
        "release": name,
        "exp_name": str(run_cfg.exp_name),
        "step": step,
        "source_checkpoint": str(source),
        "base_model": f"{run_cfg.model.base}@{run_cfg.model.get('base_revision')}" if run_cfg.finetune.load_base else None,
        "dataset": str(run_cfg.data.repo_id),
        "action_source": str(run_cfg.data.action_source),
        "weights_dtype": str(cfg.dtype),
        "exported_at": datetime.datetime.now().astimezone().isoformat(),
        "openarm_smolvla_commit": _git_commit(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log.info("release %s: %.2f GB", out, size / 1e9)

    if cfg.repo_id:
        upload(out, str(cfg.repo_id), private=bool(cfg.private))
    return out


def upload(directory: Path, repo_id: str, private: bool = True) -> None:
    """The release into <repo_id>/<release name>/ of a Hugging Face model repo."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(directory),
        path_in_repo=directory.name,
        commit_message=f"release {directory.name}",
    )
    log.info("uploaded to https://huggingface.co/%s/tree/main/%s", repo_id, directory.name)
