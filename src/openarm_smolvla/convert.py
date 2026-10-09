"""openarm_mc HDF5 episodes -> the LeRobot dataset SmolVLA trains on.

Writes HF_LEROBOT_HOME/<data.repo_id>/ (LeRobot dataset format v3) with,
per frame:

    observation.images.<camera>  uint8 [H, W, 3]  each camera data.cameras names
                                 (chest_depth encoded as ACT scales depth)
    observation.state            float32 [16]  qpos
    action                       float32 [16]  the labels (episode.py)
    task                         the prompt, from the file or the manifest

and openarm_conversion.json beside it: which episodes went in, with which
action source, and why the others did not.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import shutil
from pathlib import Path

from omegaconf import DictConfig

from openarm_smolvla import constants as C
from openarm_smolvla import paths
from openarm_smolvla.episode import EpisodeReader
from openarm_smolvla.episode import episode_id
from openarm_smolvla.episode import find_episodes
from openarm_smolvla.images import parse_image
from openarm_smolvla.manifest import Manifest

log = logging.getLogger(__name__)
REPORT = "openarm_conversion.json"


@dataclasses.dataclass
class Planned:
    path: Path
    id: str
    task: str | None = None
    skip: str | None = None  # why it is left out, or None
    image_size: tuple[int, int] | None = None  # (height, width)


def used_cameras(cfg: DictConfig) -> list[str]:
    """The recorder's cameras the dataset holds, in the order SmolVLA sees them."""
    cameras = [str(c) for c in cfg.data.cameras]
    for camera in cameras:
        if camera not in (C.CAMERA_RGB, C.CAMERA_DEPTH):
            raise ValueError(f"data.cameras: unknown camera {camera!r}; the recorder has chest, chest_depth")
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError(f"data.cameras must name each camera once, got {cameras}")
    return cameras


def plan(cfg: DictConfig) -> list[Planned]:
    """Every episode under paths.raw, with its task or the reason it is skipped."""
    raw = paths.resolve(cfg.paths.raw)
    if not raw.is_dir():
        raise FileNotFoundError(f"paths.raw: {raw} is not a directory")
    manifest_path = paths.resolve(cfg.data.manifest) if cfg.data.manifest else None
    manifest = Manifest.load(manifest_path if manifest_path and manifest_path.exists() else None)
    need_depth = C.CAMERA_DEPTH in used_cameras(cfg)

    planned = []
    for path in find_episodes(raw):
        item = Planned(path, episode_id(path, raw))
        with EpisodeReader(path) as episode:
            problems = episode.problems(int(cfg.data.min_frames), need_depth=need_depth)
            meta = manifest.resolve(item.id, episode.task)
            if not problems:
                item.image_size = episode.image_size
        item.task = meta.task
        if problems:
            item.skip = "; ".join(problems)
        elif meta.exclude:
            item.skip = "excluded by the manifest" + (f" ({meta.extra['note']})" if "note" in meta.extra else "")
        elif meta.split != "train":
            item.skip = f"split {meta.split}"
        elif not meta.task:
            item.skip = "NO TASK"
        planned.append(item)

    # One dataset, one frame size: data.image_size, else the most common one.
    size = dataset_image_size(cfg, planned)
    for item in planned:
        if item.skip is None and item.image_size != size:
            item.skip = f"frames are {item.image_size[1]}x{item.image_size[0]}, the dataset's {size[1]}x{size[0]}"
    return planned


def dataset_image_size(cfg: DictConfig, planned: list[Planned]) -> tuple[int, int] | None:
    """(height, width) every converted frame has."""
    if cfg.data.get("image_size"):
        return tuple(int(v) for v in cfg.data.image_size)
    sizes = collections.Counter(p.image_size for p in planned if p.skip is None)
    return sizes.most_common(1)[0][0] if sizes else None


def features(cfg: DictConfig, image_size: tuple[int, int]) -> dict:
    mode = "video" if cfg.data.video else "image"
    names = list(C.JOINT_NAMES)
    out = {
        C.LEROBOT_STATE: {"dtype": "float32", "shape": (C.NUM_JOINTS,), "names": names},
        C.LEROBOT_ACTION: {"dtype": "float32", "shape": (C.NUM_JOINTS,), "names": names},
    }
    for camera in used_cameras(cfg):
        out[C.LEROBOT_IMAGE_PREFIX + camera] = {
            "dtype": mode,
            "shape": (*image_size, 3),
            "names": ["height", "width", "channels"],
        }
    return out


def convert(cfg: DictConfig) -> dict:
    """Convert per cfg (conf/convert.yaml); -> the report written beside the dataset."""
    planned = plan(cfg)
    if not planned:
        raise FileNotFoundError(f"no episodes (*.hdf5) under {paths.resolve(cfg.paths.raw)}")
    untasked = [p.id for p in planned if p.skip == "NO TASK"]
    if untasked:
        raise ValueError(
            f"{len(untasked)} episodes have no task (set one in {cfg.data.manifest}): "
            + ", ".join(untasked[:10])
            + (" ..." if len(untasked) > 10 else "")
        )
    chosen = [p for p in planned if p.skip is None]
    for p in planned:
        log.info("%-48s %s", p.id, f"skip: {p.skip}" if p.skip else f"task: {p.task}")
    if not chosen:
        raise ValueError("every episode is skipped; nothing to convert")
    if cfg.dry_run:
        return {"dry_run": True, "episodes": [dataclasses.asdict(p) | {"path": str(p.path)} for p in planned]}

    # Imported here: lerobot reads HF_LEROBOT_HOME once, at import.
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    image_size = chosen[0].image_size
    root = HF_LEROBOT_HOME / str(cfg.data.repo_id)
    if root.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{root} exists; pass overwrite=true to replace it")
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=str(cfg.data.repo_id),
        fps=int(cfg.data.fps),
        robot_type="openarm_v1_bimanual",
        features=features(cfg, image_size),
        use_videos=bool(cfg.data.video),
        image_writer_threads=int(cfg.image_writer_threads),
        image_writer_processes=int(cfg.image_writer_processes),
    )
    cameras = used_cameras(cfg)
    converted = []
    try:
        for item in chosen:
            with EpisodeReader(item.path) as episode:
                labels = episode.labels(
                    str(cfg.data.action_source), int(cfg.data.fps), float(cfg.data.max_nan_fraction)
                )
                for row, index in enumerate(labels.indices):
                    frame = {
                        C.LEROBOT_STATE: labels.state[row],
                        C.LEROBOT_ACTION: labels.actions[row],
                        C.LEROBOT_TASK: item.task,
                    }
                    for camera in cameras:
                        frame[C.LEROBOT_IMAGE_PREFIX + camera] = parse_image(episode.image(camera, index))
                    dataset.add_frame(frame)
            dataset.save_episode()
            converted.append(
                {"id": item.id, "frames": len(labels.indices), "action_source": labels.source, "notes": labels.notes}
            )
            log.info("%s: %d frames, actions from %s %s", item.id, len(labels.indices), labels.source, labels.notes or "")
    finally:
        # Without it the parquet files lack their footers and the dataset cannot be read.
        dataset.finalize()

    report = {
        "repo_id": str(cfg.data.repo_id),
        "root": str(root),
        "fps": int(cfg.data.fps),
        "image_size": list(image_size),
        "action_source": str(cfg.data.action_source),
        "cameras": cameras,
        "episodes": converted,
        "skipped": [{"id": p.id, "reason": p.skip} for p in planned if p.skip],
        "frames": sum(e["frames"] for e in converted),
    }
    (root / REPORT).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
