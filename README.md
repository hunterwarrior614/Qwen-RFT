# QwenVL-RFT

QwenVL-RFT 是一个面向视觉多选问答任务的 Qwen2.5-VL LoRA 微调工程。项目以 `Qwen/Qwen2.5-VL-3B-Instruct` 为基础模型，将 Thyme-SFT 中的单图多选 VQA 样本转换为适合 Qwen-VL 训练的 JSONL 数据，并提供三条训练路线：

- SFT warm start
- PPO 强化学习微调
- GRPO 强化学习微调

当前仓库默认使用 LoRA 和 4-bit 量化加载，不会对原始基础模型权重做 full fine-tuning。模型、数据、训练输出属于本地大文件，默认不提交到仓库。

## 项目结构

```text
QwenVL-RFT/
  configs/
    sft_qwen_vl_lora.yaml
    ppo_qwen_vl_lora.yaml
    grpo_qwen_vl_lora.yaml
  scripts/
    data_process/
      convert_thyme_sft_to_qwen_vl_rl.py
      analyse_thyme_sft.ipynb
      inspect_qwen_vl_rl_data.ipynb
    train/
      train_sft_qwen_vl_lora.py
      train_ppo_qwen_vl_lora.py
      train_grpo_qwen_vl_lora.py
    plot_metrics.py
  src/qwen_vl_rl/
  tests/
  README.md
```

## 数据与模型地址

本项目依赖的数据和模型需要在本地准备，当前仓库不包含 parquet、jsonl、checkpoint 或模型权重文件。

| 类型 | 来源 | 推荐本地路径 | 说明 |
|---|---|---|---|
| 原始数据 | `Kwai-Keye/Thyme-SFT` | `../Thyme-SFT/data/wo_thinking_thyme_single_round-00000-of-00146.parquet` | Thyme-SFT 单轮 VQA parquet 文件 |
| 基础模型 | `Qwen/Qwen2.5-VL-3B-Instruct` | `../Qwen/Qwen2.5-VL-3B-Instruct` | tokenizer、processor、base model weights 的只读输入目录 |
| PPO 数据 | 本项目转换生成 | `data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_ppo.jsonl` | 使用 `messages` 字段，供 SFT/PPO 使用 |
| GRPO 数据 | 本项目转换生成 | `data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_grpo.jsonl` | 使用 `prompt` 字段，供 GRPO 使用 |
| 转换清单 | 本项目转换生成 | `data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_rl_manifest.json` | 记录输入、输出、图片格式和样本数量 |

当前数据的关键内容如下：

- 样本数：`1242`
- 任务类型：单图、多选视觉问答
- `choice_letter` 覆盖率：`100%`
- 训练目标为 `<answer>A</answer>` 形式下的 `A/B/C/D` exact match
- 转换后的 `question`、`messages` 或 `prompt` 会附带 `### Output Format` 输出格式约束
- `base_question` 保留原始题干，便于检查和报告展示
- 默认只保留第一张图片，并以 `data:image/jpeg;base64,...` 的 data URI 形式写入 JSONL

## 文件功能说明

### 配置文件

| 文件 | 功能 |
|---|---|
| `configs/sft_qwen_vl_lora.yaml` | SFT 训练配置，包括基础模型路径、训练数据、batch size、学习率、LoRA 参数、保存和评估间隔 |
| `configs/ppo_qwen_vl_lora.yaml` | PPO 训练配置，包括 SFT adapter 路径、生成参数、PPO 超参数、日志与 checkpoint 策略 |
| `configs/grpo_qwen_vl_lora.yaml` | GRPO 训练配置，包括 SFT adapter 路径、每个 prompt 的生成数量、GRPO 超参数和输出路径 |

### 脚本入口

| 文件 | 功能 |
|---|---|
| `scripts/data_process/convert_thyme_sft_to_qwen_vl_rl.py` | 将 Thyme-SFT parquet 转换为 Qwen-VL PPO/GRPO JSONL 数据，并写出 manifest |
| `scripts/data_process/analyse_thyme_sft.ipynb` | 原始 Thyme-SFT 数据分析 notebook |
| `scripts/data_process/inspect_qwen_vl_rl_data.ipynb` | 转换后 RL 数据检查 notebook |
| `scripts/train/train_sft_qwen_vl_lora.py` | SFT warm start 训练入口，支持 checkpoint 续训和训练后测试集报告生成 |
| `scripts/train/train_ppo_qwen_vl_lora.py` | PPO LoRA 训练入口，支持从 SFT/PPO checkpoint 接续训练，支持 `--test-only` |
| `scripts/train/train_grpo_qwen_vl_lora.py` | GRPO LoRA 训练入口，支持从 SFT/GRPO checkpoint 接续训练，支持 `--test-only` |
| `scripts/plot_metrics.py` | 根据 `metrics.jsonl` 单独重画训练曲线，支持 SFT、PPO、GRPO 自动识别 |

### 核心模块

| 文件 | 功能 |
|---|---|
| `src/qwen_vl_rl/answering.py` | `<answer>...</answer>` 内容提取、选项字母解析与标准答案格式化 |
| `src/qwen_vl_rl/reward.py` | 多选题 reward 计算，正确、错误有效选项、无效输出分别给分 |
| `src/qwen_vl_rl/data.py` | PPO/GRPO JSONL 数据读取、数据集划分和 Qwen-VL batch collator |
| `src/qwen_vl_rl/sft.py` | SFT 数据集、collator，以及从 PPO 格式数据构造 SFT 监督目标 |
| `src/qwen_vl_rl/ppo.py` | PPO rollout 生成、优势估计、minibatch 构造和 PPO loss 计算 |
| `src/qwen_vl_rl/grpo.py` | GRPO rollout 生成、组内 advantage 计算、minibatch 构造和 GRPO loss 计算 |
| `src/qwen_vl_rl/modeling_common.py` | dtype、BitsAndBytes 量化配置、LoRA target module 匹配等通用模型工具 |
| `src/qwen_vl_rl/modeling_ppo.py` | PPO policy/value head、reference model、LoRA policy 构建与 checkpoint 保存 |
| `src/qwen_vl_rl/config.py` | PPO/GRPO 训练配置 dataclass 与 YAML 加载 |
| `src/qwen_vl_rl/collator_utils.py` | Qwen-VL processor 输入构造、图片解码、padding side 管理和 prompt 元数据收集 |
| `src/qwen_vl_rl/reports.py` | 测试集逐样本预测记录、HTML/JSONL 报告和 accuracy 汇总 |
| `src/qwen_vl_rl/plotting.py` | 从 `metrics.jsonl` 渲染训练曲线 |
| `src/qwen_vl_rl/training_io.py` | checkpoint 查找、续训状态、optimizer/scheduler 状态保存和训练日志 |
| `src/qwen_vl_rl/utils.py` | 随机种子、路径解析、JSON 写入、图片 data URI 解码与缩放等通用工具 |

### 测试

| 目录 | 功能 |
|---|---|
| `tests/` | 覆盖数据转换、答案解析、reward、collator、PPO/GRPO 指标、报告生成、绘图和 checkpoint I/O 等逻辑 |

## 环境准备

推荐使用 Python 3.11 和 CUDA 12.4 对应的 PyTorch。以下命令以 bash 为例：

```bash
conda create -n qwen_rl python=3.11 -y
conda activate qwen_rl

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install \
  transformers==4.56.2 \
  accelerate==1.13.0 \
  trl==1.3.0 \
  peft==0.19.1 \
  datasets==4.8.5 \
  bitsandbytes==0.49.2 \
  sentencepiece \
  pyarrow \
  pyyaml \
  pillow \
  matplotlib \
  plotly \
  modelscope \
  pytest
```

训练前建议确认 GPU 可见：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

如果机器上有多张 GPU，直接运行 `python scripts/train/...py` 通常只会使用单进程的 `cuda:0`。需要多卡训练时，应使用 `accelerate launch --num_processes <N>` 启动。

## 数据准备

先下载 Thyme-SFT 原始 parquet 和 Qwen2.5-VL 基础模型：

```bash
export THYME_FILE=data/wo_thinking_thyme_single_round-00000-of-00146.parquet
export THYME_PARQUET=../Thyme-SFT/$THYME_FILE
export QWEN_MODEL_DIR=../Qwen/Qwen2.5-VL-3B-Instruct

modelscope download --dataset Kwai-Keye/Thyme-SFT "$THYME_FILE" --local_dir ../Thyme-SFT
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct --local_dir "$QWEN_MODEL_DIR"
```

然后将原始 parquet 转换为本项目使用的 RL 数据格式：

```bash
python scripts/data_process/convert_thyme_sft_to_qwen_vl_rl.py \
  --input "$THYME_PARQUET" \
  --output-dir data \
  --output-prefix wo_thinking_thyme_single_round-00000-of-00146 \
  --export-targets both \
  --image-mode first \
  --image-format data_uri
```

转换完成后应生成：

```text
data/
  wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_ppo.jsonl
  wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_grpo.jsonl
  wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_rl_manifest.json
```

## 训练流程

推荐流程是先进行 SFT warm start，再从 SFT adapter 接续做 PPO 或 GRPO。原因是当前 reward 基于选项 exact match，信号较稀疏；先让模型稳定输出 `<answer>A</answer>` 格式，后续 RL 训练会更稳定。

### SFT

单卡 smoke test：

```bash
python scripts/train/train_sft_qwen_vl_lora.py \
  --config configs/sft_qwen_vl_lora.yaml \
  --max-steps 2
```

正式训练时去掉 `--max-steps`：

```bash
python scripts/train/train_sft_qwen_vl_lora.py \
  --config configs/sft_qwen_vl_lora.yaml
```

多卡训练示例：

```bash
accelerate launch --num_processes 4 scripts/train/train_sft_qwen_vl_lora.py \
  --config configs/sft_qwen_vl_lora.yaml
```

### PPO

训练前先确认 `configs/ppo_qwen_vl_lora.yaml` 中的 `model.sft_adapter_path` 指向可用的 SFT adapter，例如：

```yaml
model:
  sft_adapter_path: outputs/sft/default/checkpoint-100/adapter
```

单卡训练：

```bash
python scripts/train/train_ppo_qwen_vl_lora.py \
  --config configs/ppo_qwen_vl_lora.yaml
```

多卡训练：

```bash
accelerate launch --num_processes 4 scripts/train/train_ppo_qwen_vl_lora.py \
  --config configs/ppo_qwen_vl_lora.yaml
```

### GRPO

训练前先确认 `configs/grpo_qwen_vl_lora.yaml` 中的 `model.sft_adapter_path` 指向可用的 SFT adapter。

单卡训练：

```bash
python scripts/train/train_grpo_qwen_vl_lora.py \
  --config configs/grpo_qwen_vl_lora.yaml
```

多卡训练：

```bash
accelerate launch --num_processes 4 scripts/train/train_grpo_qwen_vl_lora.py \
  --config configs/grpo_qwen_vl_lora.yaml
```

## 续训与测试

三种训练入口都支持从已有 checkpoint 继续训练：

```bash
python scripts/train/train_sft_qwen_vl_lora.py \
  --config configs/sft_qwen_vl_lora.yaml \
  --resume-from-checkpoint latest

python scripts/train/train_ppo_qwen_vl_lora.py \
  --config configs/ppo_qwen_vl_lora.yaml \
  --resume-from-checkpoint outputs/ppo/default/checkpoint-100 \
  --max-steps 150

python scripts/train/train_grpo_qwen_vl_lora.py \
  --config configs/grpo_qwen_vl_lora.yaml \
  --resume-from-checkpoint outputs/grpo/default/checkpoint-50 \
  --max-steps 100
```

`--resume-from-checkpoint` 可以传 `latest`、`checkpoint-XX`、完整 checkpoint 路径，或 checkpoint 下的 `adapter/` 路径。续训时会保留原有 `metrics.jsonl` 并继续追加；`--max-steps` 表示目标总 step 数，例如从 `checkpoint-100` 继续到 `--max-steps 150` 会再训练 50 个 step。

PPO 和 GRPO 支持只运行测试集：

```bash
python scripts/train/train_ppo_qwen_vl_lora.py \
  --config configs/ppo_qwen_vl_lora.yaml \
  --test-only

python scripts/train/train_grpo_qwen_vl_lora.py \
  --config configs/grpo_qwen_vl_lora.yaml \
  --test-only
```

PPO 测试时会优先加载 `outputs/ppo/<run_name>/checkpoint-*/adapter` 中最新的 PPO checkpoint。也可以显式指定 adapter：

```bash
python scripts/train/train_ppo_qwen_vl_lora.py \
  --config configs/ppo_qwen_vl_lora.yaml \
  --test-only \
  --policy-adapter-path <checkpoint_dir_or_adapter_dir>
```

## 输出与结果查看

训练输出默认写入：

```text
outputs/
  sft/default/
  ppo/default/
  grpo/default/
```

常见输出文件包括：

- `metrics.jsonl`：训练和评估指标日志
- `training_curve.png`：训练曲线
- `train_summary.json`：训练摘要
- `checkpoint-*/adapter/`：LoRA adapter
- `checkpoint-*/processor/`：processor 保存目录
- `test_results.jsonl`：测试集逐样本预测结果
- `test_results.html`：带图片、题目、预测和答案的可视化测试报告

已有 `metrics.jsonl` 时，可以单独重画训练曲线：

```bash
python scripts/plot_metrics.py outputs/sft/default --kind sft
python scripts/plot_metrics.py outputs/ppo/default --kind ppo --rolling-window 20
python scripts/plot_metrics.py outputs/grpo/default --kind grpo --rolling-window 20
```

默认输出到对应 run 目录下的 `training_curve.png`，也可以通过 `--output <path>` 指定图片路径。

主要关注指标：

- SFT：`eval_loss`、`eval_exact_match`
- PPO/GRPO：`reward_mean`、`accuracy`、`valid_option_rate`、`response_length_mean`、`kl_mean`
- 测试报告：`test_results.html` 中的逐样本预测、答案和图片对照

配置中的 `eval_size` 表示 validation split，`test_size` 对应最终逐样本报告使用的测试集。报告中的 `raw_response` 是模型原始 decoded response，`prediction` 是去掉首尾空白后的版本。

## 当前默认配置

### SFT

- 配置文件：`configs/sft_qwen_vl_lora.yaml`
- 输出目录：`outputs/sft/default`
- 基础模型：`../Qwen/Qwen2.5-VL-3B-Instruct`
- 训练文件：`data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_ppo.jsonl`
- 数据划分：`train_size=1000`、`eval_size=121`、`test_size=121`
- 训练参数：`per_device_train_batch_size=2`、`gradient_accumulation_steps=2`、`learning_rate=2.0e-4`、`num_train_epochs=2`
- 模型设置：`load_in_4bit=true`、`gradient_checkpointing=true`、`torch_dtype=bfloat16`、`attn_implementation=sdpa`
- LoRA：`r=16`、`alpha=32`、`dropout=0.05`

### PPO

- 配置文件：`configs/ppo_qwen_vl_lora.yaml`
- 输出目录：`outputs/ppo/default`
- 训练文件：`data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_ppo.jsonl`
- SFT adapter：`outputs/sft/default/checkpoint-100/adapter`
- 数据划分：`train_size=1000`、`eval_size=121`、`test_size=121`
- 训练轮数：`num_train_epochs=30`
- 生成参数：`max_new_tokens=16`、`temperature=0.7`、`top_p=0.9`
- PPO 参数：`per_device_prompt_batch_size=1`、`per_device_minibatch_size=1`、`ppo_epochs=2`、`cliprange=0.2`、`kl_coef=0.02`
- 模型设置：`load_in_4bit=true`、`gradient_checkpointing=true`

### GRPO

- 配置文件：`configs/grpo_qwen_vl_lora.yaml`
- 输出目录：`outputs/grpo/default`
- 训练文件：`data/wo_thinking_thyme_single_round-00000-of-00146.qwen_vl_grpo.jsonl`
- SFT adapter：`outputs/sft/default/checkpoint-100/adapter`
- 数据划分：`train_size=1000`、`eval_size=121`、`test_size=121`
- 训练轮数：`num_train_epochs=20`
- 生成参数：`max_new_tokens=16`、`temperature=0.7`、`top_p=0.9`
- GRPO 参数：`per_device_prompt_batch_size=1`、`num_generations=4`、`per_device_minibatch_size=1`、`grpo_epochs=1`、`cliprange=0.2`、`kl_coef=0.02`
- 模型设置：`load_in_4bit=true`、`gradient_checkpointing=true`

## 运行测试

仓库内的单元测试主要覆盖不依赖真实大模型权重的本地逻辑：

```bash
pytest
```

如果只想验证某一部分，可以指定测试文件，例如：

```bash
pytest tests/test_convert_thyme_sft_to_qwen_vl_rl.py
pytest tests/test_reward.py
pytest tests/test_training_io.py
```
