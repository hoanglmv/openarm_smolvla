# openarm_smolvla

Fine-tune [SmolVLA](https://huggingface.co/lerobot/smolvla_base) on teleoperation
episodes of the **OpenArm V1** bimanual arm, and serve it to the robot.

- **Input:** one RGB-D image from the chest camera and the 16 joint positions.
- **Output:** the next **50 joint states** (1 s at 50 Hz) for all 16 joints, as absolute targets.
- **One command** goes from the Hugging Face dataset to a trained model.

```
HF dataset (HDF5) ─► LeRobot dataset ─► norm stats ─► SmolVLA training ─► checkpoint ─► release
                                                                                           │ serve (GPU)
robot ◄── [50, 16] joint targets ◄── websocket ◄── SmolVLAPolicy client ◄──────────────────┘
```

## Contents

- [Model](#model)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Where everything is saved](#where-everything-is-saved)
- [Configuration](#configuration)
- [Step by step](#step-by-step)
- [Evaluation and export](#evaluation-and-export)
- [Serving to the robot](#serving-to-the-robot)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Project layout](#project-layout)

## Model

| | |
|---|---|
| Network | SmolVLA (~450M parameters): SmolVLM2-500M vision-language backbone and a flow-matching action expert, initialised from `lerobot/smolvla_base` |
| Image | the chest RGB image with depth as a **4th channel**; the vision encoder's patch embedding is widened from 3 to 4 channels and trained |
| Depth | millimetres → metres, clipped to 0.2–1.2 m, scaled to 0..1 |
| State | 16 joints: `left_j1..j7, left_gripper, right_j1..j7, right_gripper` (rad; grippers as finger stroke in m), z-score normalised |
| Output | `[50, 16]` absolute joint targets in the same order and units; row 0 is the step the observation was taken at |
| Labels | the recorded joint states, `action[t : t + 50]`, padded and masked at the end of an episode |
| Prompt | the episode's task from [manifests/episodes.yaml](manifests/episodes.yaml) |
| Training | action expert, projections and the 4-channel patch embedding train; SigLIP and the language model stay frozen |
| Validation | 15 % of the episodes held out, picked by a hash so the choice is stable as episodes are added |

The design and its alternatives (depth as a separate image, RGB only,
teleop commands as labels, delta actions) are in [docs/DESIGN.md](docs/DESIGN.md).

## Requirements

**Training machine**

- Linux, an NVIDIA GPU, and **driver 580 or newer** (the locked PyTorch 2.11 is built for CUDA 13.0; `nvidia-smi` shows the driver)
- ~40 GB free disk for the 30 GB of episodes and the converted dataset, plus room for checkpoints
- Internet access to Hugging Face (dataset, base model) and, optionally, Weights & Biases
- `curl` or `wget`; everything else ([uv](https://docs.astral.sh/uv/), Python 3.12, PyTorch, LeRobot 0.6.1) is installed by the script
- Optional: `ffmpeg` (`sudo apt install ffmpeg`) for faster video decoding

**Robot computer:** Python ≥ 3.10 and the small client package in [client/](client/) (numpy, msgpack, websockets; no PyTorch).

## Quick start

```bash
git clone https://github.com/hoanglmv/openarm_smolvla.git
cd openarm_smolvla
scripts/smoke_train.sh                     # 15-30 min: every step on 3 episodes, on the GPU
scripts/run_pipeline.sh exp_name=v1        # the real training
```

[scripts/smoke_train.sh](scripts/smoke_train.sh) runs the whole chain small
before the long run: setup, download, conversion, loading the base model,
30 training steps at the real batch size (an out-of-memory error shows
here), checkpoints, resume, offline evaluation, export, and a real frame
served to the robot's client. Its outputs are named `smoke` and do not mix
with real runs; the episodes it downloads are reused.

The first run creates `.env` and stops at the Weights & Biases check. Put
your keys in it and run the same command again:

```bash
# .env (git-ignored; .env.example is the template)
WANDB_API_KEY=...        # https://wandb.ai/authorize
HF_TOKEN=hf_...          # https://huggingface.co/settings/tokens, read access; faster downloads
```

[scripts/run_pipeline.sh](scripts/run_pipeline.sh) installs uv if needed,
runs `uv sync`, checks that PyTorch can use the GPU, then runs
[scripts/train_pipeline.py](scripts/train_pipeline.py):

| Step | What happens | When it is skipped |
|---|---|---|
| 1. wandb | checks the API key before anything long starts | `wandb.enable=false` |
| 2. download | fetches [Tuyen062004/data_openarm_8_10](https://huggingface.co/datasets/Tuyen062004/data_openarm_8_10) (85 episodes, 30 GB) after checking the disk has room | files already on disk |
| 3. convert | HDF5 → LeRobot dataset (MP4 video, parquet) | the dataset already holds exactly these episodes and settings |
| 4. norm stats | mean and std of state and actions (seconds) | never |
| 5. train | 20 000 steps, batch 64 | |

Every step does only what is missing, so the same command resumes after an
interruption. Add `resume=true` to continue the training run itself.

Long runs belong in `tmux` (`tmux new -s smolvla`; `Ctrl+B D` to detach,
`tmux attach -t smolvla` to return). On a machine with several GPUs, pick
one:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 scripts/run_pipeline.sh exp_name=v1
```

## Where everything is saved

All paths are inside the repository and git-ignored, except the caches.

| What | Where |
|---|---|
| Python environment | `.venv/` |
| Keys | `.env` |
| Downloaded episodes | `data/raw/data_openarm_8_10/*.hdf5` |
| LeRobot dataset | `data/lerobot/openarm/openarm_rgbd_qpos_50hz/` (+ `openarm_conversion.json`: episodes used and skipped) |
| Norm stats | `assets/h50_absolute/openarm/openarm_rgbd_qpos_50hz/stats.json` |
| Checkpoints | `checkpoints/smolvla_expert/<exp_name>/checkpoints/<step>/`; `checkpoints/last` points at the newest |
| wandb local files | `checkpoints/smolvla_expert/<exp_name>/wandb/` |
| Releases | `releases/<name>/` |
| Logs of each command | `outputs/<script>/<date_time>/` |
| Base model, tokenizer | `~/.cache/huggingface/` |

A checkpoint holds `pretrained_model/` (weights, `config.json`, the
processors with their norm stats, `openarm_config.yaml`: what serving and
evaluation load) and `training_state/` (optimizer and scheduler, for
`resume=true`). Every checkpoint is kept; budget a few GB each.

Other locations: `paths.raw=`, `paths.lerobot_home=`, `paths.assets=`,
`paths.checkpoints=`, `paths.releases=`.

## Configuration

Settings live in [conf/](conf/) ([Hydra](https://hydra.cc)) and are
overridden on the command line as `key=value`. The same overrides reach
every step of the pipeline.

| Group | Options | |
|---|---|---|
| `data` | **`openarm_rgbd`**, `openarm_rgb`, `openarm_rgb_depth` | one 4-channel RGB-D image; RGB only; depth as a second gray image |
| `finetune` | **`expert`**, `full`, `dummy` | action expert only; every weight; a small random network for CPU checks |
| `model` | **`smolvla`** | base checkpoint and revision, chunk size, denoising steps |
| `experiment` | `smoke` | a 10-step CPU run of the whole pipeline |

Common options:

| Option | Default | |
|---|---|---|
| `exp_name` | date and time | the run's name: checkpoint folder and wandb run |
| `batch_size` | 64 | lower it if the GPU runs out of memory |
| `num_train_steps` | 20000 | |
| `save_interval` | 2000 | steps between checkpoints |
| `lr.peak_lr` | 1e-4 | cosine schedule, 1000 warm-up steps |
| `resume` | false | continue `exp_name` from its last checkpoint |
| `device` | auto | `cuda` when available |
| `data.action_source` | qpos | `commands`: the teleop targets instead of the recorded joint states |
| `data.delta_actions` | false | arm joints relative to the current state; grippers stay absolute |
| `data.val_fraction` | 0.15 | share of episodes held out |
| `data.depth_init` | zero | the depth channel's initial weights: `zero`, or `mean` of the RGB weights |
| `download.limit` | all | download only the first N episodes |
| `data.max_episodes` | all | train on only the first N episodes under `data/raw/` |
| `download.enable` | true | false: use only what is in `data/raw/` |
| `wandb.enable` / `wandb.mode` | true / online | `offline` logs locally (`uv run wandb sync` later) |
| `wandb.upload_checkpoints` | false | also upload every checkpoint to wandb |

Examples:

```bash
scripts/run_pipeline.sh exp_name=try download.limit=10 data.max_episodes=10 num_train_steps=2000   # a short run
scripts/run_pipeline.sh exp_name=v1 batch_size=32                              # less GPU memory
scripts/run_pipeline.sh exp_name=v1 resume=true                                # continue v1
scripts/run_pipeline.sh exp_name=full_v1 finetune=full                         # train every weight
scripts/run_pipeline.sh exp_name=cmd_v1 data.action_source=commands            # teleop targets as labels
```

Each run's resolved configuration is saved with every checkpoint.

## Step by step

The pipeline's steps also run on their own (`uv run` uses the project's environment):

```bash
uv run scripts/download_dataset.py                  # episodes -> data/raw/
uv run scripts/inspect_dataset.py                   # hours per task, skipped episodes, label quality
uv run scripts/convert_dataset.py                   # -> data/lerobot/ (overwrite=true to redo)
uv run scripts/compute_norm_stats.py                # -> assets/
uv run scripts/train.py exp_name=v1                 # -> checkpoints/
```

**Tasks.** SmolVLA reads a task as its prompt. Set one for every episode in
[manifests/episodes.yaml](manifests/episodes.yaml); the conversion refuses
episodes without one. The same file excludes episodes (`exclude: true`) and
holds out others (`split: val`):

```yaml
defaults:
  task: null
rules:                                   # glob on the episode id, later rules win
  - match: "data_openarm_8_10/*"
    task: "pick up the green box and place it on the white paper"
  - match: "data_openarm_8_10/episode_20261008_1529*"
    exclude: true
    note: "camera stalled"
```

An episode id is its path under `data/raw/` without `.hdf5`.

**Image size.** The model is trained on the episodes' frame size and the
server reports it to the robot, which must stream the same size.
`data_openarm_8_10` is **640×480**. Do not mix frame sizes in one dataset;
the conversion keeps the most common one and skips the others.

## Evaluation and export

**Offline evaluation** predicts a chunk every 10 frames of each held-out
episode and compares it with the recorded trajectory, next to a baseline
that holds the arm still. A useful model beats the baseline clearly.

```bash
uv run scripts/offline_eval.py checkpoint=checkpoints/smolvla_expert/v1                       # newest step
uv run scripts/offline_eval.py checkpoint=checkpoints/smolvla_expert/v1/checkpoints/020000
```

It prints the arm RMSE (rad) and gripper MAE (mm), overall and along the
chunk, and writes the full report beside the checkpoint.

**Export** the checkpoint you keep into a self-contained release, without
the optimizer state, and with the tokenizer files so it serves offline:

```bash
uv run scripts/export_policy.py checkpoint=checkpoints/smolvla_expert/v1/checkpoints/020000
uv run scripts/export_policy.py checkpoint=... dtype=bfloat16                  # half the size
uv run scripts/export_policy.py checkpoint=... repo_id=<user>/openarm-smolvla  # also upload to Hugging Face
```

## Serving to the robot

**On the GPU machine**, serve a release or a checkpoint over websocket:

```bash
uv run scripts/serve.py checkpoint=releases/smolvla_expert_v1_020000 port=8000
```

**On the robot computer**, install the client:

```bash
python3 -m pip install --user ./client
```

The client is a policy class with the interface a robot policy node calls:

```python
from openarm_smolvla_client.policy import SmolVLAPolicy

policy = SmolVLAPolicy(checkpoint="ws://<gpu-host>:8000")
actions = policy.predict(rgb, depth, qpos)
#   rgb    uint8   [H, W, 3]
#   depth  uint16  [H, W]   millimetres, 0 = no reading
#   qpos   float32 [16]
#   ->     float32 [50, 16] absolute joint targets
```

A ROS policy node that loads its policy by import path takes
`policy:=openarm_smolvla_client.policy:SmolVLAPolicy checkpoint:=ws://<gpu-host>:8000`.
On connecting, the client reads the model's chunk size, frame size and
cameras from the server, and sends raw frames; depth encoding, resizing and
normalisation happen on the server, in the code that trained the model.
Targets are clipped to the OpenArm V1 joint limits.

| Variable (robot computer) | |
|---|---|
| `OPENARM_SMOLVLA_PROMPT` | the instruction; otherwise the server's default (the dataset's task) |
| `OPENARM_SMOLVLA_SERVER` | the server, when `checkpoint` is not given |
| `OPENARM_SMOLVLA_CLIP=0` | do not clip to the joint limits |

The model predicts one second at a time. Executing about half of each chunk
before asking for the next (a new chunk every 0.5 s) is a good starting
point. The server answers `GET /healthz` for monitoring.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `there is a GPU but torch cannot use it` | the NVIDIA driver is older than 580; update it |
| `CUDA out of memory` | `batch_size=32` (or lower), with `resume=true` to keep the progress |
| download at a few hundred kB/s | put `HF_TOKEN` in `.env`; `download.max_workers=8` |
| `... GB to download but ... GB free` | free space, or `paths.raw=/bigger/disk/raw` |
| `wandb is on but has no API key` | `WANDB_API_KEY` in `.env`, or `wandb.enable=false` |
| `N episodes have no task` | give them one in `manifests/episodes.yaml` |
| `... exists; pass overwrite=true or resume=true` | the run name is taken: another `exp_name`, `resume=true`, or `overwrite=true` |
| ROS's Python on `PYTHONPATH` breaks the environment | `env -u PYTHONPATH uv run ...` (the pipeline script does this) |
| slow data loading, `pyav` in the log | `sudo apt install ffmpeg` |

## Development

```bash
uv run pytest                          # CPU, ~15 min; builds and serves small random models
scripts/smoke_train.sh                 # GPU, every step with the real model on 3 episodes
uv run scripts/train_pipeline.py experiment=smoke download.limit=2   # the whole pipeline on a CPU, 10 steps
```

`experiment=smoke` trains a small random SmolVLA (`finetune=dummy`): the
real data, processors, checkpointing and serving, but nothing learned.

## Project layout

```
conf/                    Hydra configs: pipeline, train, convert, inspect, eval, export, serve + groups
manifests/episodes.yaml  task, exclusions and held-out episodes
scripts/                 run_pipeline.sh and one entry point per step
src/openarm_smolvla/
  episode.py manifest.py        reading episodes, labels, tasks and splits
  convert.py images.py          HDF5 -> LeRobot dataset; image and depth encoding
  norm_stats.py                 normalisation statistics
  train_config.py run.py        Hydra -> LeRobot configs; training
  *_smolvla_rgbd.py             SmolVLA with a 4-channel RGB-D input
  processors.py                 delta-action steps
  policy.py server.py           inference and the websocket server
  evaluate.py export.py         offline evaluation; releases
client/                  openarm_smolvla_client: the robot-side policy (Python >= 3.10)
docs/DESIGN.md           design notes
tests/
```
