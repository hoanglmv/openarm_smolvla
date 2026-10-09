# openarm_smolvla

Fine-tune **SmolVLA** ([LeRobot](https://github.com/huggingface/lerobot),
`lerobot/smolvla_base`) on episodes recorded with [openarm_mc](../openarm_mc),
and run it on the OpenArm V1 bimanual arm through openarm_mc's `openarm_act`,
the same way ACT runs today. The sibling of
[openarm_pizero](../openarm_pizero) (π0 / π0.5): same data, same manifest,
same labels, same robot-side contract, so the three models compare directly.

```
openarm_mc recorder ──► HDF5 ──► convert ──► LeRobot v3 ──► norm stats ──► train ──► checkpoint
                                                                                        │ serve (GPU)
openarm_act (ROS) ◄── [50,16] absolute targets ◄── websocket ◄──────────────────────────┘
   policy:=openarm_smolvla_client.policy:SmolVLAPolicy
```

The design and the reasons behind it: [docs/DESIGN.md](docs/DESIGN.md).

## Same inputs and outputs as ACT

| | ACT (openarm_mc) | SmolVLA (this repo) |
|---|---|---|
| `predict(...)` | `rgb` uint8 [H,W,3], `depth` uint16 mm, `qpos` [16] | the same, plus a prompt |
| returns | [50, 16] absolute targets, 50 Hz | the same |
| joints | `left_j1..j7, left_gripper, right_j1..j7, right_gripper`; rad, gripper stroke m | the same |
| depth | 4th image channel, 0.2–1.2 m → 0..1 | `data=openarm_rgbd`: the same scaling, as a gray second image. `data=openarm` (default): RGB only |

The client clips the targets to V1's joint limits.

## Setup

**GPU machine** (training and serving). Python 3.12 comes from uv.

```bash
git clone <this repo> && cd openarm_smolvla
uv sync                                  # LeRobot 0.6.1 (PyTorch), Hydra, h5py
uv run pytest                            # the pipeline on fake episodes, CPU
sudo apt install ffmpeg                  # video decoding with torchcodec; without it, PyAV (slower)
```

LeRobot is pinned to one release (0.6.1); its dataset format and processor
API change between minor versions. `lerobot/smolvla_base` is pinned to a
Hub revision in [conf/model/smolvla.yaml](conf/model/smolvla.yaml).

On a machine where ROS is sourced, `PYTHONPATH` points at ROS's Python 3.10
packages. Drop it for this environment: `env -u PYTHONPATH uv run ...`.

**Robot PC** (ROS 2 Humble, `/usr/bin/python3`): only the client, which
needs numpy, msgpack and websockets, no torch.

```bash
/usr/bin/python3 -m pip install --user ./client
```

## Configuration (Hydra)

Everything lives in [conf/](conf/) and is overridden on the command line:

| Group | Options | |
|---|---|---|
| `model` | `smolvla` | base checkpoint, chunk size, denoising steps, image size |
| `finetune` | `expert`, `full`, `dummy` | action expert only (LeRobot's recipe), every weight, a small random net for CPU checks |
| `data` | `openarm`, `openarm_rgbd` | cameras, labels, fps, delta actions of the dataset |
| `experiment` | `smoke` | presets |
| `paths` | `default` | `raw`, `lerobot_home`, `assets`, `checkpoints`, `releases` |

Training, norm stats and conversion compose the same `data` file, so they
cannot disagree about the dataset. Every checkpoint carries the run's
resolved config (`pretrained_model/openarm_config.yaml`) and its processors
(prompt tokenizer, norm stats, delta steps). Serving and evaluation rebuild
the model from those, so they need only the checkpoint path.

## Workflow

**1. Bring the episodes over**, from Hugging Face or from the robot PC:

```bash
uv run scripts/download_dataset.py                 # Tuyen062004/data_openarm_8_10 -> data/raw/data_openarm_8_10/
uv run scripts/download_dataset.py limit=2         # the first two episodes, for a quick run
rsync -a robot-pc:openarm_mc/dataset/ data/raw/    # or straight from the recorder
```

Episode ids are paths under `data/raw/` without `.hdf5`, e.g.
`data_openarm_8_10/episode_20261008_152119`.

**2. Give every episode a task** in [manifests/episodes.yaml](manifests/episodes.yaml)
(the same file format as openarm_pizero's), together with the episodes to
exclude and the ones to hold out (`split: val`). The recorder does not store
a task yet. SmolVLA reads the task as its prompt.

**3. Inspect** before converting: hours per task, skipped episodes, and
whether `joint_commands` can serve as the labels:

```bash
uv run scripts/inspect_dataset.py
```

**4. Convert**, then compute norm stats (seconds; they read the parquet
files, not the videos):

```bash
uv run scripts/convert_dataset.py                    # data=openarm
uv run scripts/compute_norm_stats.py
```

**5. Train:**

```bash
uv run scripts/train.py exp_name=v1                              # finetune=expert, batch 64, 20k steps
uv run scripts/train.py finetune=full exp_name=v1 batch_size=32
uv run scripts/train.py exp_name=v1 resume=true                  # continue from the last checkpoint
uv run scripts/train.py exp_name=v1 data.delta_actions=true      # after compute_norm_stats.py data.delta_actions=true
```

Checkpoints land in `checkpoints/smolvla_<finetune>/<exp_name>/checkpoints/<step>/`
(LeRobot's layout: `pretrained_model/` and `training_state/`), with
`checkpoints/last` pointing at the newest.

Training logs to **Weights & Biases**, project `openarm_smolvla`, one run
named `smolvla_<finetune>_<exp_name>`. Paste the key from
https://wandb.ai/authorize into `.env` (git-ignored; `.env.example` is the
template), or run `uv run wandb login` once. Training stops at the start
when wandb is on and no key is found.

```bash
uv run scripts/train.py exp_name=v1 wandb.entity=<team>        # another entity than WANDB_ENTITY
uv run scripts/train.py exp_name=v1 wandb.mode=offline         # log locally; `uv run wandb sync` later
uv run scripts/train.py exp_name=v1 wandb.enable=false
```

`resume=true` continues the same wandb run. Checkpoints are not uploaded
to wandb (`wandb.upload_checkpoints=true` to upload each one, ~1–2 GB).

**6. Evaluate offline** on the held-out episodes, against a "hold still"
baseline:

```bash
uv run scripts/offline_eval.py checkpoint=checkpoints/smolvla_expert/v1/checkpoints/020000
```

**7. Export the checkpoint you keep:**

```bash
uv run scripts/export_policy.py checkpoint=checkpoints/smolvla_expert/v1/checkpoints/020000
uv run scripts/export_policy.py checkpoint=... repo_id=<user>/openarm-smolvla   # + upload
```

| | Training checkpoint | Release (`releases/<name>/`) |
|---|---|---|
| for | resuming training | serving, archiving, sharing |
| holds | `pretrained_model/` (float32 weights, processors, config), `training_state/` (optimizer) | the weights, processors, `openarm_config.yaml`, `vlm/`, `metadata.json` |
| Hub | needs SmolVLM2's config and tokenizer from the Hub (cached after the first time) | serves offline: `vlm/` holds SmolVLM2's config and tokenizer |

`dtype=bfloat16` halves the weights; they are loaded back into float32.
`metadata.json` records the model, base revision, dataset, frame size, step
and commit.

**8. Serve on the GPU machine, run on the robot:**

```bash
# GPU machine
uv run scripts/serve.py checkpoint=releases/smolvla_expert_v1_020000 port=8000

# robot PC (openarm_mc workspace sourced)
OPENARM_SMOLVLA_PROMPT="pick up the green box and place it on the white paper" \
ros2 launch openarm_act act.launch.py \
    policy:=openarm_smolvla_client.policy:SmolVLAPolicy \
    checkpoint:=ws://<gpu-host>:8000
```

The dashboard works as it does for ACT. Generate trajectory and Start
preview the chunk first, and nothing moves until Execute. Like π0, SmolVLA
is meant to run a chunk partway and replan, rather than ensembling every
step. A starting point is:

```bash
ros2 param set /openarm_act temporal_ensemble false
ros2 param set /openarm_act query_period 25        # a new chunk every 0.5 s
```

| Variable (robot PC) | |
|---|---|
| `OPENARM_SMOLVLA_PROMPT` | the instruction; otherwise the server's default (the dataset's task when it has one) |
| `OPENARM_SMOLVLA_SERVER` | the server, when `checkpoint:=` is not given |
| `OPENARM_SMOLVLA_CLIP=0` | do not clip to V1's joint limits |

The server speaks openpi's websocket protocol (msgpack with NumPy arrays,
metadata on connect, `GET /healthz`), the one openarm_pizero serves.

## Checking the pipeline without a GPU

```bash
uv run pytest                                  # fake episodes, CPU; includes a small SmolVLA saved, loaded, served
uv run scripts/download_dataset.py limit=2
uv run scripts/convert_dataset.py overwrite=true
uv run scripts/compute_norm_stats.py
uv run scripts/train.py experiment=smoke       # 10 steps of a small random SmolVLA on the CPU
uv run scripts/export_policy.py checkpoint=checkpoints/smolvla_dummy/smoke
uv run scripts/serve.py checkpoint=releases/smolvla_dummy_smoke_000010 device=cpu
```

`experiment=smoke` trains `finetune=dummy`: SmolVLM2's architecture with
random weights, its language model cut to two layers, for 10 steps. It runs
the real data, processors, checkpointing and serving, but learns nothing.
SmolVLM2's config and tokenizer come from the Hub the first time.

## Image size

A model's frames have to match the camera's at inference. `data.image_size`
defaults to the episodes' own size, is recorded into the run, and is served
to the client. openarm_act then refuses a camera that gives anything else.
`data_openarm_8_10` (recorded on 2026-10-08) is **640×480**. openarm_mc has
recorded 424×240 since 2026-10-09, and those frames show a different field
of view, so do not mix the two sizes in one dataset. SmolVLA itself
letterboxes every frame to 512×512.

## Layout

```
conf/                    Hydra: train / convert / inspect / serve / eval / export + groups
manifests/               task, exclusions, held-out split per episode
src/openarm_smolvla/     episode reading and labels, conversion, images, norm stats,
                         Hydra -> LeRobot configs, delta-action steps, training,
                         inference (policy.py), websocket server, eval, export
scripts/                 the Hydra entry points above
client/                  openarm_smolvla_client: SmolVLAPolicy for openarm_act (Python 3.10)
data/ assets/ checkpoints/ releases/   (git-ignored) episodes, norm stats, runs, exports
tests/
```
