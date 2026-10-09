"""Hydra -> LeRobot configs, images, conversion, norm stats, delta steps,
a small SmolVLA through save -> load -> infer, and the client against the server."""

import contextlib
import json
import socket
import threading
import time
import uuid

import numpy as np
import pytest
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf

from openarm_smolvla import constants as C
from openarm_smolvla import paths


def compose_cfg(name="train", overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(paths.CONF_DIR)):
        return compose(config_name=name, overrides=list(overrides))


# ------------------------------------------------------------------ config


@pytest.mark.parametrize("finetune", ["expert", "full", "dummy"])
@pytest.mark.parametrize("data", ["openarm_rgb", "openarm_rgb_depth", "openarm_rgbd"])
def test_every_combination_composes(finetune, data):
    from openarm_smolvla.train_config import policy_overrides

    cfg = compose_cfg(overrides=[f"finetune={finetune}", f"data={data}", "exp_name=t", "device=cpu"])
    assert cfg.name == f"smolvla_{finetune}"
    labels = "qpos" if data == "openarm_rgbd" else "commands"
    assert cfg.data.repo_id == f"openarm/{data}_{labels}_50hz"
    assert list(cfg.data.cameras) == (["chest"] if data == "openarm_rgb" else ["chest", "chest_depth"])
    overrides = policy_overrides(cfg)
    assert overrides["chunk_size"] == 50 and overrides["device"] == "cpu"
    assert overrides["train_expert_only"] == (finetune == "expert")
    assert overrides["freeze_vision_encoder"] == (finetune == "expert")
    assert not overrides["load_vlm_weights"]  # the SmolVLA checkpoint carries the VLM
    assert overrides["input_features"] == {}  # taken from the dataset


def test_dummy_builds_a_train_config(tmp_path):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    from openarm_smolvla.train_config import build_train_config

    cfg = compose_cfg(overrides=["experiment=smoke"])
    train_cfg = build_train_config(cfg, tmp_path / "run")
    assert isinstance(train_cfg.policy, SmolVLAConfig)
    assert train_cfg.policy.pretrained_path is None and train_cfg.policy.num_vlm_layers == 2
    assert train_cfg.steps == 10 and train_cfg.batch_size == 2
    assert train_cfg.scheduler.num_decay_steps == 10
    assert train_cfg.dataset.repo_id == "openarm/openarm_rgbd_qpos_50hz"  # the default data


def test_norm_stats_shared_across_finetuning():
    from openarm_smolvla.norm_stats import norm_stats_path

    expert = compose_cfg(overrides=["finetune=expert"])
    full = compose_cfg(overrides=["finetune=full"])
    assert norm_stats_path(expert) == norm_stats_path(full)
    assert norm_stats_path(compose_cfg(overrides=["data.delta_actions=true"])) != norm_stats_path(expert)
    assert norm_stats_path(compose_cfg(overrides=["data.fps=25"])) != norm_stats_path(expert)


def test_smoke_experiment_overrides():
    cfg = compose_cfg(overrides=["experiment=smoke"])
    assert cfg.finetune.name == "dummy"
    assert cfg.num_train_steps == 10
    assert not cfg.wandb.enable


def test_dotenv_and_the_wandb_key_check(tmp_path, monkeypatch):
    from openarm_smolvla.run import check_wandb

    env = tmp_path / ".env"
    env.write_text("# comment\nWANDB_API_KEY='abc123'\nWANDB_ENTITY=\nexport OPENARM_TEST_SET=shell\n")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("OPENARM_TEST_SET", "shell wins")
    assert paths.load_dotenv(env) == ["WANDB_API_KEY"]
    assert __import__("os").environ["WANDB_API_KEY"] == "abc123"
    assert __import__("os").environ["OPENARM_TEST_SET"] == "shell wins"
    check_wandb(compose_cfg())  # the key is there

    monkeypatch.delenv("WANDB_API_KEY")
    monkeypatch.setenv("NETRC", str(tmp_path / "no_netrc"))
    with pytest.raises(RuntimeError, match="no API key"):
        check_wandb(compose_cfg())
    check_wandb(compose_cfg(overrides=["wandb.mode=offline"]))
    check_wandb(compose_cfg(overrides=["experiment=smoke"]))  # wandb off


# ------------------------------------------------------------------ images


def test_depth_encoding_is_acts():
    from openarm_smolvla.images import depth_to_image

    depth = np.array([[0, 200, 700, 1200, 5000]], dtype=np.uint16)
    image = depth_to_image(depth)
    assert image.shape == (1, 5, 3) and image.dtype == np.uint8
    # ACT: clip(m, 0.2, 1.2) - 0.2, over 1.0 -> 0, 0, 0.5, 1, 1
    np.testing.assert_array_equal(image[0, :, 0], [0, 0, 128, 255, 255])


def test_image_layouts_agree():
    from openarm_smolvla.images import parse_image
    from openarm_smolvla.images import to_model_input

    rgb = np.random.default_rng(0).integers(0, 256, (240, 424, 3), dtype=np.uint8)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0  # as LeRobot decodes
    np.testing.assert_array_equal(parse_image(chw), rgb)
    np.testing.assert_allclose(to_model_input(rgb), chw)
    assert parse_image(np.zeros((240, 424), dtype=np.uint16)).shape == (240, 424, 3)


# -------------------------------------------------------------- conversion


def _convert(tmp_path, episode_factory, data="openarm_rgb", extra=()):
    from openarm_smolvla.convert import convert

    episode_factory("2026-10-08/episode_a", n=80, task="pick the cup")
    episode_factory("2026-10-08/episode_b", n=60, seed=1, commands="no_grippers")
    episode_factory("2026-10-09/episode_c", n=60)
    manifest = tmp_path / "episodes.yaml"
    manifest.write_text("defaults: {task: default task}\nrules:\n  - {match: '2026-10-09/*', split: val}\n")
    overrides = [
        f"paths.raw={tmp_path / 'raw'}",
        f"paths.assets={tmp_path / 'assets'}",
        f"paths.checkpoints={tmp_path / 'checkpoints'}",
        f"data={data}",
        f"data.manifest={manifest}",
        f"data.name=test_{uuid.uuid4().hex[:12]}",  # one dataset per test, in the session's HF_LEROBOT_HOME
        *extra,
    ]
    return convert(compose_cfg("convert", overrides)), overrides


@pytest.mark.parametrize("data", ["openarm_rgb", "openarm_rgb_depth", "openarm_rgbd"])
def test_convert_and_read_back(tmp_path, episode_factory, data):
    from lerobot.datasets import LeRobotDataset

    from openarm_smolvla.episode import EpisodeReader
    from openarm_smolvla.images import depth_to_image

    report, overrides = _convert(tmp_path, episode_factory, data, ["data.video=false", "data.val_fraction=0"])
    assert [e["id"] for e in report["episodes"]] == ["2026-10-08/episode_a", "2026-10-08/episode_b"]
    assert report["skipped"] == [{"id": "2026-10-09/episode_c", "reason": "split val"}]
    cfg = compose_cfg(overrides=[*overrides, "data.val_fraction=0"])
    source = str(cfg.data.action_source)  # openarm_rgbd: qpos, the others: commands
    # commands: episode_b's grippers fall back to next_qpos and lose the last row
    frames = 80 + (60 if source == "qpos" else 59)
    assert report["frames"] == frames

    dataset = LeRobotDataset(str(cfg.data.repo_id), delta_timestamps={"action": [i / 50 for i in range(50)]})
    assert len(dataset) == frames and dataset.num_episodes == 2
    sample = dataset[0]
    assert sample["task"] == "pick the cup"
    first = tmp_path / "raw/2026-10-08/episode_a.hdf5"
    with EpisodeReader(first) as ep:
        labels = ep.labels(source)
        rgb0, depth0 = ep.rgb(0), ep.depth(0)
    np.testing.assert_allclose(np.asarray(sample["observation.state"]), labels.state[0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(sample["action"]), labels.actions[:50], atol=1e-6)
    image = (np.asarray(sample["observation.images.chest"]) * 255).round().astype(np.uint8).transpose(1, 2, 0)
    np.testing.assert_array_equal(image, rgb0)  # PNG frames are lossless
    if data != "openarm_rgb":
        depth = np.asarray(sample["observation.images.chest_depth"])
        np.testing.assert_array_equal((depth * 255).round().astype(np.uint8).transpose(1, 2, 0), depth_to_image(depth0))


def test_convert_reuses_a_current_dataset_only(tmp_path, episode_factory):
    from openarm_smolvla.convert import convert

    first, overrides = _convert(tmp_path, episode_factory, extra=["data.video=false"])
    again = compose_cfg("convert", [*overrides, "data.video=false", "reuse_current=true"])
    assert convert(again).get("reused")
    with pytest.raises(FileExistsError):  # neither flag: refuse
        convert(compose_cfg("convert", [*overrides, "data.video=false"]))
    episode_factory("2026-10-08/episode_d", n=60, seed=2)  # a new episode arrives
    report = convert(again)
    assert not report.get("reused") and len(report["episodes"]) == 3
    (tmp_path / "episodes.yaml").write_text("defaults: {task: another task}\n")  # a task is reworded
    report = convert(again)
    assert not report.get("reused") and report["episodes"][1]["task"] == "another task"
    assert convert(again).get("reused")


def test_convert_refuses_episodes_without_a_task(tmp_path, episode_factory):
    from openarm_smolvla.convert import convert

    episode_factory("2026-10-08/episode_a")
    cfg = compose_cfg("convert", [f"paths.raw={tmp_path / 'raw'}", f"data.manifest={tmp_path / 'none.yaml'}"])
    with pytest.raises(ValueError, match="no task"):
        convert(cfg)


@pytest.mark.parametrize("delta", [False, True])
def test_norm_stats_cover_whole_chunks(tmp_path, episode_factory, delta):
    from openarm_smolvla.norm_stats import compute_norm_stats
    from openarm_smolvla.norm_stats import read_episodes

    _, overrides = _convert(tmp_path, episode_factory, extra=["data.video=false"])
    cfg = compose_cfg(overrides=[*overrides, f"data.delta_actions={str(delta).lower()}", "model.chunk_size=10"])
    stats = json.loads(compute_norm_stats(cfg).read_text())
    episodes = read_episodes(_root(cfg))
    targets = []
    for state, action in episodes:
        for t in range(len(action)):
            for k in range(10):
                if t + k < len(action):
                    target = action[t + k].copy()
                    if delta:
                        target[list(C.ARM_JOINTS)] -= state[t][list(C.ARM_JOINTS)]
                    targets.append(target)
    targets = np.asarray(targets)
    np.testing.assert_allclose(stats["action"]["mean"], targets.mean(0), atol=1e-6)
    np.testing.assert_allclose(stats["action"]["std"], targets.std(0), atol=1e-6)
    states = np.concatenate([s for s, _ in episodes])
    np.testing.assert_allclose(stats["observation.state"]["mean"], states.mean(0), atol=1e-6)


def _root(cfg):
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / str(cfg.data.repo_id)


# ------------------------------------------------------------ delta steps


def test_delta_steps_round_trip():
    import torch
    from lerobot.processor import NormalizerProcessorStep
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor import UnnormalizerProcessorStep
    from lerobot.processor.converters import batch_to_transition
    from lerobot.processor.converters import policy_action_to_transition
    from lerobot.processor.converters import transition_to_batch
    from lerobot.processor.converters import transition_to_policy_action

    from openarm_smolvla.processors import AbsoluteActionsStep
    from openarm_smolvla.processors import DeltaActionsStep
    from openarm_smolvla.processors import add_delta_steps
    from openarm_smolvla.processors import link_delta_steps

    pre = PolicyProcessorPipeline(
        steps=[NormalizerProcessorStep(features={}, norm_map={})],
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    post = PolicyProcessorPipeline(
        steps=[UnnormalizerProcessorStep(features={}, norm_map={})],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    add_delta_steps(pre, post)
    assert isinstance(pre.steps[0], DeltaActionsStep) and isinstance(post.steps[1], AbsoluteActionsStep)

    rng = torch.Generator().manual_seed(0)
    state = torch.randn(4, 1, 16, generator=rng)  # SmolVLA's batches: one observation step
    chunk = torch.randn(4, 50, 16, generator=rng)
    out = pre({C.LEROBOT_STATE: state, C.LEROBOT_ACTION: chunk.clone()})
    arm, grippers = list(C.ARM_JOINTS), list(C.GRIPPERS)
    torch.testing.assert_close(out[C.LEROBOT_ACTION][..., arm], chunk[..., arm] - state[:, :, arm])
    torch.testing.assert_close(out[C.LEROBOT_ACTION][..., grippers], chunk[..., grippers])
    torch.testing.assert_close(post(out[C.LEROBOT_ACTION]), chunk)

    # Serialised and loaded back, the link is gone until link_delta_steps.
    loaded = AbsoluteActionsStep(**post.steps[1].get_config())
    post.steps[1] = loaded
    with pytest.raises(RuntimeError, match="not linked"):
        post(out[C.LEROBOT_ACTION])
    link_delta_steps(pre, post)
    torch.testing.assert_close(post(out[C.LEROBOT_ACTION]), chunk)


# ------------------------------------------------- a small model, end to end


def _save_dummy_checkpoint(tmp_path, episode_factory, delta: bool, data: str = "openarm_rgb"):
    """A random 2-layer SmolVLA saved the way LeRobot's training saves one."""
    import torch
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies import make_policy
    from lerobot.policies import make_pre_post_processors

    from openarm_smolvla.norm_stats import compute_norm_stats
    from openarm_smolvla.norm_stats import load_norm_stats
    from openarm_smolvla.processors import add_delta_steps
    from openarm_smolvla.run import with_stats
    from openarm_smolvla.train_config import build_policy_config
    from openarm_smolvla.train_config import dataset_image_size
    from openarm_smolvla.train_config import dataset_tasks
    from openarm_smolvla.train_config import save_run_config

    _, overrides = _convert(tmp_path, episode_factory, data, extra=["data.video=false"])
    cfg = compose_cfg(overrides=[*overrides, "experiment=smoke", f"data.delta_actions={str(delta).lower()}",
                                 "model.resize_imgs_with_padding=[128,128]", "model.num_steps=2"])
    stats = load_norm_stats(compute_norm_stats(cfg))
    meta = LeRobotDatasetMetadata(str(cfg.data.repo_id))
    torch.manual_seed(0)
    policy = make_policy(build_policy_config(cfg), ds_meta=meta)
    pre, post = make_pre_post_processors(policy.config, dataset_stats=with_stats(meta.stats, stats))
    if delta:
        add_delta_steps(pre, post)
    out = tmp_path / "checkpoints/run/checkpoints/000010/pretrained_model"
    policy.save_pretrained(out)
    (out.parents[1] / "last").symlink_to("000010")  # as LeRobot's update_last_checkpoint
    pre.save_pretrained(out)
    post.save_pretrained(out)
    cfg.data.image_size, cfg.data.tasks = dataset_image_size(cfg), dataset_tasks(cfg)
    save_run_config(cfg, out)
    return out, cfg


@pytest.mark.parametrize("delta", [False, True])
def test_saved_model_serves_through_openarm_policy(tmp_path, episode_factory, vlm_files, delta):
    import torch

    from openarm_smolvla.policy import OpenArmPolicy

    out, cfg = _save_dummy_checkpoint(tmp_path, episode_factory, delta)
    policy = OpenArmPolicy(out.parents[2], device="cpu")  # the run directory: its newest step
    assert policy.directory == out.resolve()
    assert policy.metadata["step"] == 10 and policy.metadata["image_size"] == [240, 424]
    assert sorted(cfg.data.tasks) == ["default task", "pick the cup"]
    assert policy.default_prompt is None  # two tasks: the client has to say which

    rng = np.random.default_rng(0)
    obs = {
        "images": {"chest": rng.integers(0, 256, (240, 424, 3), dtype=np.uint8)},
        "state": rng.normal(size=16).astype(np.float32),
        "prompt": "pick the cup",
    }
    torch.manual_seed(1)
    actions = policy.infer(obs)["actions"]
    assert actions.shape == (50, 16) and actions.dtype == np.float32 and np.all(np.isfinite(actions))

    # By hand: the same noise, the normaliser's stats, delta or not.
    torch.manual_seed(1)
    with torch.inference_mode():
        batch = policy.preprocessor(policy.batch(obs))
        raw = policy.policy.predict_action_chunk(batch)[0]
    stats = json.loads((paths.resolve(cfg.paths.assets) / f"h50_{'delta' if delta else 'absolute'}"
                        / str(cfg.data.repo_id) / "stats.json").read_text())["action"]
    expected = raw.numpy() * np.asarray(stats["std"]) + np.asarray(stats["mean"])
    if delta:
        expected[:, list(C.ARM_JOINTS)] += obs["state"][list(C.ARM_JOINTS)]
    np.testing.assert_allclose(actions, expected, rtol=1e-4, atol=1e-4)

    with pytest.raises(ValueError, match="no prompt"):
        policy.infer({k: v for k, v in obs.items() if k != "prompt"})


def test_export_then_serve_offline(tmp_path, episode_factory, vlm_files, monkeypatch):
    import torch

    from openarm_smolvla.export import export
    from openarm_smolvla.policy import OpenArmPolicy

    out, cfg = _save_dummy_checkpoint(tmp_path, episode_factory, delta=False)
    export_cfg = compose_cfg("export", [f"checkpoint={out.parents[1]}", f"paths.releases={tmp_path / 'releases'}",
                                        "dtype=bfloat16"])
    release = export(export_cfg)
    assert release.name == "smolvla_dummy_smoke_000010"
    metadata = json.loads((release / "metadata.json").read_text())
    assert metadata["step"] == 10 and metadata["weights_dtype"] == "bfloat16"
    assert (release / "vlm" / "tokenizer.json").exists() and not list(release.glob("vlm/*.safetensors"))
    from safetensors import safe_open

    with safe_open(release / "model.safetensors", "pt") as weights:
        assert {weights.get_slice(k).get_dtype() for k in weights.keys()} == {"BF16"}

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # everything from the release itself
    policy = OpenArmPolicy(release, device="cpu", default_prompt_override="pick the cup")
    obs = {"images": {"chest": np.zeros((240, 424, 3), dtype=np.uint8)}, "state": np.zeros(16, dtype=np.float32)}
    torch.manual_seed(0)
    assert policy.infer(obs)["actions"].shape == (50, 16)
    assert OmegaConf.load(release / paths.RUN_CONFIG).name == "smolvla_dummy"


# --------------------------------------------------- ACT's RGB-D input


def test_held_out_is_stable_and_sized():
    from openarm_smolvla.convert import held_out

    ids = [f"day/episode_{i:03d}" for i in range(85)]
    held = held_out(ids, 0.15)
    assert len(held) == 13 and held == held_out(list(reversed(ids)), 0.15)
    assert len(held - held_out(ids + ["day/episode_new"], 0.15)) <= 1  # a new episode moves one at most
    assert held != held_out(ids, 0.15, seed=1)
    assert held_out(ids, 0.0) == set() and held_out(ids[:1], 0.5) == set()


def test_act_config_holds_out_and_labels_like_act(tmp_path, episode_factory):
    report, overrides = _convert(tmp_path, episode_factory, "openarm_rgbd",
                                 ["data.video=false", "data.val_fraction=0.5"])
    cfg = compose_cfg("convert", [*overrides, "data=openarm_rgbd"])
    assert cfg.data.action_source == "qpos" and cfg.data.depth_as == "channel"
    reasons = [s["reason"] for s in report["skipped"]]
    assert "split val" in reasons and any(r.startswith("split val (val_fraction") for r in reasons)
    assert len(report["episodes"]) == 1
    assert report["episodes"][0]["action_source"] == "qpos"  # action[t] = qpos[t], as act_pipeline


@pytest.mark.parametrize("depth_init", ["zero", "mean"])
def test_rgbd_patch_embedding_widens_3_channel_weights(tmp_path, vlm_files, depth_init):
    import torch
    from lerobot.configs import FeatureType
    from lerobot.configs import PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from openarm_smolvla.configuration_smolvla_rgbd import SmolVLARGBDConfig
    from openarm_smolvla.modeling_smolvla_rgbd import SmolVLARGBDPolicy

    features = {
        "input_features": {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
            "observation.images.chest": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.chest_depth": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
    }
    small = {"num_vlm_layers": 1, "load_vlm_weights": False, "device": "cpu", "train_expert_only": True,
             "freeze_vision_encoder": True, "resize_imgs_with_padding": (64, 64)}
    torch.manual_seed(0)
    base = SmolVLAPolicy(SmolVLAConfig(**small, **features))  # 3 channels, as smolvla_base
    base.save_pretrained(tmp_path / "base")
    rgb_weight = base.model.vlm_with_expert.get_vlm_model().vision_model.embeddings.patch_embedding.weight

    config = SmolVLARGBDConfig(**small, **features, depth_init=depth_init)
    policy = SmolVLARGBDPolicy.from_pretrained(tmp_path / "base", config=config)
    vision = policy.model.vlm_with_expert.get_vlm_model().vision_model
    weight = vision.embeddings.patch_embedding.weight
    assert weight.shape[1] == 4 and weight.requires_grad  # trainable, as ACT's conv1_adapter
    assert not vision.encoder.layers[0].self_attn.q_proj.weight.requires_grad  # the rest frozen
    torch.testing.assert_close(weight[:, :3], rgb_weight)
    expected = rgb_weight.mean(1) if depth_init == "mean" else torch.zeros_like(rgb_weight[:, 0])
    torch.testing.assert_close(weight[:, 3], expected)

    # Depth goes in as the RGB image's 4th channel.
    rgb, depth = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    images, masks = policy.prepare_images({"observation.images.chest": rgb, "observation.images.chest_depth": depth})
    assert len(images) == 1 and images[0].shape == (2, 4, 64, 64)
    torch.testing.assert_close(images[0][:, 3], depth[:, 0] * 2 - 1)


def test_rgbd_model_serves_like_act(tmp_path, episode_factory, vlm_files):
    """The client's raw RGB + depth + qpos in, [50, 16] absolute joint states out."""
    import torch

    from openarm_smolvla.policy import OpenArmPolicy

    out, cfg = _save_dummy_checkpoint(tmp_path, episode_factory, delta=False, data="openarm_rgbd")
    assert json.loads((out / "config.json").read_text())["type"] == "smolvla_rgbd"
    policy = OpenArmPolicy(out, device="cpu")
    assert type(policy.policy).__name__ == "SmolVLARGBDPolicy"
    assert policy.metadata["depth_as"] == "channel" and policy.metadata["cameras"] == ["chest", "chest_depth"]

    rng = np.random.default_rng(0)
    obs = {"images": {"chest": rng.integers(0, 256, (240, 424, 3), dtype=np.uint8),
                      "chest_depth": rng.integers(0, 1500, (240, 424), dtype=np.uint16)},
           "state": rng.normal(size=16).astype(np.float32), "prompt": "pick the cup"}
    torch.manual_seed(0)
    actions = policy.infer(obs)["actions"]
    assert actions.shape == (50, 16) and np.all(np.isfinite(actions))

    # Depth reaches the network once its weights are not zero.
    with torch.no_grad():
        policy.policy.model.vlm_with_expert.get_vlm_model().vision_model.embeddings.patch_embedding.weight[:, 3] += 1.0
    flat = dict(obs, images=dict(obs["images"], chest_depth=np.full((240, 424), 300, np.uint16)))
    torch.manual_seed(0)
    a = policy.infer(obs)["actions"]
    torch.manual_seed(0)
    b = policy.infer(flat)["actions"]
    assert not np.allclose(a, b)
    with pytest.raises(KeyError, match="chest_depth"):
        policy.infer(dict(obs, images={"chest": obs["images"]["chest"]}))


# ------------------------------------------------------------------ client


def _free_port():
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _EchoPolicy:
    """Stands in for OpenArmPolicy: remembers the observation, returns a
    chunk that runs past V1's limits."""

    def __init__(self):
        self.seen = None

    def infer(self, obs):
        self.seen = obs
        actions = np.repeat(np.asarray(obs["state"], dtype=np.float32)[None], 50, axis=0)
        actions[:, 3] = 5.0  # left_j4 beyond 2.44 rad
        return {"actions": actions}


def _serve(policy, metadata):
    from openarm_smolvla.server import PolicyServer

    port = _free_port()
    server = PolicyServer(policy, host="127.0.0.1", port=port, metadata=metadata)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return port


def _connect(port, **kwargs):
    from openarm_smolvla_client.policy import SmolVLAPolicy

    for _ in range(50):
        try:
            return SmolVLAPolicy(checkpoint=f"ws://127.0.0.1:{port}", **kwargs)
        except ConnectionError:
            time.sleep(0.1)
    raise AssertionError("the server never came up")


@pytest.mark.parametrize("data", ["openarm_rgb", "openarm_rgb_depth", "openarm_rgbd"])
def test_client_against_a_server(data):
    from openarm_smolvla.train_config import policy_metadata

    cfg = compose_cfg(overrides=[f"data={data}", "data.default_prompt=hand it over", "data.image_size=[240,424]"])
    policy = _EchoPolicy()
    client = _connect(_serve(policy, policy_metadata(cfg)))
    assert client.chunk_size == 50 and client.image_size == (240, 424)
    assert client.prompt == "hand it over"

    rgb = np.zeros((240, 424, 3), dtype=np.uint8)
    depth = np.full((240, 424), 600, dtype=np.uint16)
    qpos = np.zeros(16, dtype=np.float32)
    actions = client.predict(rgb, depth, qpos)
    assert actions.shape == (50, 16)
    assert np.all(actions[:, 3] <= 2.443461 + 1e-6)  # clipped to V1
    assert ("chest_depth" in policy.seen["images"]) == (data != "openarm_rgb")
    assert policy.seen["images"]["chest"].dtype == np.uint8 and policy.seen["prompt"] == "hand it over"
    assert client.server_ms is not None
    assert "SmolVLAPolicy" in client.describe()


def test_client_sees_server_errors_and_reconnects():
    from openarm_smolvla.train_config import policy_metadata

    class Failing(_EchoPolicy):
        calls = 0

        def infer(self, obs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("first call fails")
            return super().infer(obs)

    cfg = compose_cfg(overrides=["data.default_prompt=x", "data.image_size=[240,424]"])
    client = _connect(_serve(Failing(), policy_metadata(cfg)))
    args = (np.zeros((240, 424, 3), np.uint8), np.zeros((240, 424), np.uint16), np.zeros(16, np.float32))
    with pytest.raises(RuntimeError, match="first call fails"):
        client.predict(*args)
    assert client.predict(*args).shape == (50, 16)  # a new connection


def test_client_refuses_quickly_without_a_server():
    from openarm_smolvla_client.policy import SmolVLAPolicy
    from openarm_smolvla_client.policy import parse_server

    assert parse_server("gpu:9000") == ("gpu", 9000)
    assert parse_server("ws://gpu") == ("gpu", 8000)
    with pytest.raises(ConnectionError):
        SmolVLAPolicy(checkpoint=f"127.0.0.1:{_free_port()}", connect_timeout=1.0)


def test_msgpack_matches_openpi_encoding():
    from openarm_smolvla_client import msgpack_numpy

    data = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.uint16(7), "s": "text"}
    back = msgpack_numpy.unpackb(msgpack_numpy.packb(data))
    np.testing.assert_array_equal(back["a"], data["a"])
    assert back["b"] == 7 and back["b"].dtype == np.uint16 and back["s"] == "text"
    with pytest.raises(ValueError):
        msgpack_numpy.packb({"x": np.array([object()])})
