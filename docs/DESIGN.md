# openarm_smolvla — Thiết kế

Fine-tune **SmolVLA** (Hugging Face, `lerobot/smolvla_base`) trên dữ liệu
teleop của OpenArm bimanual, rồi chạy trên robot qua đúng stack mà ACT và
π0/π0.5 ([openarm_pizero](../../openarm_pizero/docs/DESIGN.md)) đang dùng.

> Trạng thái: code đầy đủ từ dữ liệu tới robot. LeRobot pin ở **0.6.1**
> (3/8/2026), `smolvla_base` pin ở revision `5e8d12a`. Đã chạy thông trên
> CPU: convert 2 episode thật → norm stats → train 10 bước (model dummy,
> delta actions) → resume → export → serve → client. Chưa train thật trên GPU.

---

## 1. Bài toán

Giống hệt openarm_pizero (mục 1 bên đó): 16 DoF, một camera ngực RGB-D,
ghi 50 Hz, HDF5 của `openarm_ui/data_recorder.py`, không có prompt trong
file, và `action == qpos` nên nhãn phải lấy từ `joint_commands`. Toàn bộ
phần đọc episode, manifest, chọn nhãn (`commands` / `next_qpos` / `qpos`)
được chép nguyên từ openarm_pizero (`episode.py`, `manifest.py`,
`inspect_data.py`, `download.py`), nên ba model dùng **cùng một tập nhãn**.

Phía robot cũng giữ nguyên hợp đồng của `openarm_act`:

```python
policy.predict(rgb: uint8[H,W,3], depth: uint16[H,W], qpos: float32[16])
    -> float32[50, 16]      # target khớp tuyệt đối
```

## 2. SmolVLA so với π0 / π0.5

| | SmolVLA | π0 / π0.5 (openarm_pizero) |
|---|---|---|
| Kích thước | ~450M (SmolVLM2-500M cắt còn 16 lớp + action expert) | ~3.3B |
| Framework | LeRobot (PyTorch) | openpi (JAX) |
| Fine-tune nhẹ | chỉ action expert (công thức của LeRobot) | LoRA |
| Ảnh | số camera tuỳ ý, letterbox 512×512 | 3 slot cố định, 224×224 |
| Pretrain | dữ liệu cộng đồng SO-100, action tuyệt đối | dữ liệu PI, delta cho ALOHA |
| Suy luận | 10 bước flow matching | 10 bước flow matching |

SmolVLA nhỏ hơn π0 gần 8 lần: train nhanh hơn, cần GPU nhỏ hơn, suy luận
nhanh hơn. Đổi lại, nó được pretrain trên tay đơn SO-100 rẻ tiền, xa
OpenArm hai tay hơn dữ liệu của PI. So sánh ba model trên cùng dữ liệu là
mục đích chính của repo này.

## 3. Kiến trúc tổng thể

Giữ hai máy như openarm_pizero (robot PC không có GPU, ROS Humble chạy
Python 3.10, còn LeRobot 0.6.1 cần Python ≥ 3.12):

```
 Robot PC (Py 3.10, ROS 2)                    GPU server (Py 3.12, LeRobot 0.6.1)
 openarm_ui recorder ── rsync / HF ─────────► data/raw/*.hdf5
                                                 │ convert_dataset.py
                                                 ▼
                                              data/lerobot/<repo_id>/   (LeRobot v3, mp4)
                                                 │ compute_norm_stats.py
                                                 ▼
                                              assets/h50_<absolute|delta>/<repo_id>/stats.json
                                                 │ train.py  (lerobot_train.train)
                                                 ▼
                                              checkpoints/smolvla_<ft>/<exp>/checkpoints/<step>/
                                                 │ export_policy.py → releases/<name>/
                                                 │ serve.py
 openarm_act                                     ▼
   policy:=openarm_smolvla_client.policy:SmolVLAPolicy ◄──► websocket :8000
```

Server nói **giao thức của openpi** (msgpack + NumPy, gửi metadata khi kết
nối, `GET /healthz`). Module mã hoá (`client/openarm_smolvla_client/msgpack_numpy.py`)
dùng chung cho cả hai đầu, và khớp từng byte với openpi.

## 4. Dùng LeRobot thế nào

Không gọi CLI `lerobot-train` (draccus). Giống cách openarm_pizero dựng
`TrainConfig` của openpi, `train_config.py` dựng `TrainPipelineConfig` và
`SmolVLAConfig` từ cây Hydra, rồi `run.py` gọi thẳng
`lerobot.scripts.lerobot_train.train(cfg)`. Ba điểm được chèn vào bằng
`mock.patch` quanh lời gọi đó, không fork LeRobot:

1. **Norm stats** của repo này thay stats per-frame của dataset trong
   normalizer (mục 5.3).
2. **Bước delta actions** chèn vào processors khi `data.delta_actions=true`.
3. **`openarm_config.yaml`** ghi vào `pretrained_model/` của mỗi checkpoint.

Vài lựa chọn khi dựng config:

- `load_vlm_weights=False`: checkpoint SmolVLA đã chứa toàn bộ trọng số,
  kể cả VLM. Dựng SmolVLM2 từ config rồi nạp checkpoint, nên không cần tải
  trọng số SmolVLM2 riêng (chỉ config + tokenizer, vài MB).
- `input_features={}`: lấy feature từ dataset (`observation.images.chest`,
  state 16 chiều) thay vì `camera1..3`/6 chiều của SO-100. Không cần
  `rename_map`. Tên camera không ảnh hưởng trọng số, chỉ thứ tự ảnh.
- Optimizer/scheduler đặt tường minh từ Hydra (`use_policy_training_preset=False`),
  mặc định bằng preset của SmolVLA: AdamW lr 1e-4, β (0.9, 0.95), cosine
  với warmup 1000 bước, clip grad 10.
- `_resolve_pretrained_from_cli` của LeRobot đọc `sys.argv`, nên bị tắt.
  Đường dẫn base model và checkpoint resume do `run.py` tự xác định.

## 5. Map dữ liệu OpenArm sang SmolVLA

### 5.1 Bảng ánh xạ

| SmolVLA nhận | Từ OpenArm | Ghi chú |
|---|---|---|
| `observation.images.chest` | `chest_rgb` | float [3,H,W] 0..1, model tự letterbox về 512×512 |
| `observation.images.chest_depth` | depth (chỉ `data=openarm_rgbd`) | ảnh xám, mã hoá như ACT (5.4) |
| `observation.state` | `qpos` [16] | model pad lên 32 |
| `action` | nhãn [50,16] | tuyệt đối (mặc định) hoặc delta (5.2) |
| `task` | prompt từ manifest | SmolVLA thêm `\n` rồi tokenize, tối đa 48 token |

### 5.2 Delta actions: mặc định tắt

openarm_pizero bật delta (theo config ALOHA của openpi). Ở đây mặc định
**tắt**, vì `smolvla_base` được pretrain với target tuyệt đối và công thức
fine-tune của LeRobot cũng giữ tuyệt đối. `data.delta_actions=true` bật
cùng phép biến đổi như openpi: 14 khớp tay trừ đi state đầu chunk, 2
gripper giữ tuyệt đối.

LeRobot 0.6.1 có sẵn `RelativeActionsProcessorStep`, nhưng nó giả định
state [B, D], trong khi batch của SmolVLA có state [B, 1, D] (một bước quan
sát) và broadcast sai. Vì vậy `processors.py` có cặp bước riêng,
`openarm_delta_actions` (trước normalizer) và `openarm_absolute_actions`
(sau unnormalizer). Chúng được đăng ký vào registry của LeRobot, nên được
lưu vào `policy_preprocessor.json` / `policy_postprocessor.json` và nạp lại
cùng checkpoint. Liên kết giữa hai bước (state đang xử lý) không lưu được,
nên `link_delta_steps` nối lại sau khi nạp.

### 5.3 Norm stats

SmolVLA chuẩn hoá state và action bằng mean/std. Stats per-frame của LeRobot
(`meta/stats.json`) đúng cho action tuyệt đối, nhưng sai cho delta: độ lệch
`action[t+k] − state[t]` tăng theo k. `norm_stats.py` tính stats trên mọi
cặp (t, k) của mọi chunk (cắt ở cuối episode, như loss được mask), cho cả
hai chế độ, chỉ đọc parquet nên mất vài giây. Kết quả nằm ở
`assets/h<chunk>_<absolute|delta>/<repo_id>/stats.json`, dùng chung cho mọi
chế độ fine-tune trên cùng dữ liệu.

### 5.4 Depth

SigLIP chỉ nhận RGB. `data=openarm_rgbd` thêm depth thành ảnh thứ hai,
mã hoá **đúng như ACT** (kẹp 0.2–1.2 m, scale 0..255, 3 kênh xám) bằng một
hàm duy nhất (`images.depth_to_image`), dùng cả lúc convert lẫn lúc serve.
SmolVLA nhận số camera tuỳ ý, nên không phải mượn slot cổ tay như π0.

### 5.5 Tần số, chunk, prompt

50 Hz, `chunk_size=50` (1 s) như ACT và π0. Prompt: manifest như
openarm_pizero. Run lưu danh sách task của dataset; khi dataset chỉ có một
task, server dùng nó làm prompt mặc định.

## 6. Các chế độ fine-tune

| `finetune` | Train | Ghi chú |
|---|---|---|
| `expert` (mặc định) | action expert + projection state/action | công thức của LeRobot cho `smolvla_base`, VLM đóng băng |
| `full` | mọi trọng số | bộ nhớ optimizer lớn hơn nhiều; thử sau khi có baseline `expert` |
| `dummy` | SmolVLA ngẫu nhiên, VLM cắt còn 2 lớp | chỉ để kiểm pipeline trên CPU (`experiment=smoke`) |

Không có LoRA: với `expert`, SmolVLM2 đã đóng băng và chỉ còn action
expert được train. LeRobot 0.6.1 có PEFT cho SmolVLA nếu sau này cần. Số
tham số train được in ra ở đầu mỗi lần train (`num_learnable_params`).

## 7. Checkpoint, release, serving

- Checkpoint theo layout của LeRobot: `<step>/pretrained_model/`
  (trọng số, `config.json`, processors kèm stats, `openarm_config.yaml`) và
  `<step>/training_state/` (optimizer, scheduler, RNG, để resume).
- `serve.py`, `offline_eval.py`, `export_policy.py` nhận một release, một
  `pretrained_model/`, một thư mục step, hoặc thư mục run (lấy step mới nhất).
- Release (`export_policy.py`) bỏ optimizer, tuỳ chọn ép trọng số về bf16,
  và gói config + tokenizer của SmolVLM2 vào `vlm/`, nên serve được khi
  không có mạng.
- `OpenArmPolicy` (`policy.py`) là lớp suy luận dùng chung cho server và
  offline eval: ảnh qua `images.py`, rồi đúng processors đã lưu trong
  checkpoint, rồi `predict_action_chunk`.

## 8. Đánh giá

Như openarm_pizero: offline eval trên các episode `split: val`, so với
baseline "đứng yên"; noise của flow matching được cố định theo `seed` để
so sánh công bằng giữa các checkpoint. Trên robot: cùng task, cùng tư thế
đầu, N = 20 lần mỗi model, so ACT, π0, π0.5 và SmolVLA.

## 9. Rủi ro và câu hỏi mở

- **Khoảng cách pretrain.** SO-100 là tay 6 DoF đơn, OpenArm là 2 × 8.
  State/action 16 chiều nằm gọn trong 32 chiều pad, nhưng phân bố khác hẳn.
  Cần so `expert` với `full`.
- **Một camera ngực**, giống mọi model khác trong dự án.
- **Delta hay tuyệt đối**: cần offline eval cho cả hai.
- **Bộ nhớ GPU và thời gian train**: chưa đo trên GPU. Batch 64 là mặc
  định của LeRobot; giảm `batch_size` nếu thiếu bộ nhớ.
- **Checkpoint lớn**: mỗi checkpoint giữ lại cả optimizer; `save_interval`
  2000 trên 20k bước tạo 10 checkpoint, cần chừa dung lượng đĩa.
