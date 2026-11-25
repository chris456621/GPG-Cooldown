# GPG 原版 vs. 改良版 對照報告（Train 200 / Eval 100 / G=4 / 200 Steps）

1. **基礎模型**：`Qwen2.5-3B-Instruct`（`name_or_path=models/Qwen2.5-3B-Instruct`, `hf_repo_id=Qwen/Qwen2.5-3B-Instruct` | `dtype=bfloat16` (`bf16=True`, `fp16=False`), `attn=sdpa` | LoRA: `enabled=True`, `r=8`, `alpha=16`, `dropout=0.05`, `target_modules=['q_proj', 'v_proj']` | 4-bit: `load_in_4bit=False`）
2. **硬體與環境**：GPU=`NVIDIA GeForce RTX 3060` | backend=`cuda` | VRAM=`12.0 GB` (free `10.99 GB`), `CC=[8, 6]`, `profile=vram_12gb` | OS=`Windows 11` | Python=`3.11.9` | PyTorch=`2.5.1+cu124` (CUDA `12.4`) | Config SHA256=`1a778a57729e`
3. **評測方式**：Greedy Decoding（`do_sample=False`, `max_new_tokens=256`, `batch_size=8`）
4. **重要參數**：`Train=200` / `Eval=100` / `G=4` / `Steps=200` | `L=192`, `max_prompt_length=256`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=2`, `learning_rate=5e-05`, `lr_scheduler_type=cosine`, `warmup_ratio=0.1`, `cooldown_steps=20`, `cooldown_backoff=False`, `cooldown_backoff_factor=2.0`, `cooldown_max_steps=0`, `sample_strategy=uniform`, `scale_rewards=False`, `min_inverse_alpha=0.0`, `seed=42`

## 1. 訓練過程指標比較

| 組別 | 耗時 (s) | 全程平均 Reward | 前 1/4 Reward | 後 1/4 Reward | 平均全對題/步 (`easy`) | 有效梯度比 (`inv_alpha`) | 非零梯度步數 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPG 原版 (Uniform) | 11087.5 | 0.2194 | 0.1075 | 0.3050 | 0.090 | 0.268 | 197/200 |
| GPG + Cooldown (uniform, cd=20, backoff=False) | 7869.7 | 0.1313 | 0.0950 | 0.1150 | 0.022 | 0.250 | 194/200 |

## 2. 確定性評測結果（Train-200 留存率 & Eval-100 泛化力）

| 模型版本 | Train-200 留存率（答對/總數） | Eval-100 泛化準確率（答對/總數） |
|---|---:|---:|
| Base (Qwen2.5-3B-Instruct) | 38.0% (76/200) | 33.0% (33/100) |
| GPG 原版 (Uniform) | 39.5% (79/200) | 36.0% (36/100) |
| GPG + Cooldown (uniform, cd=20, backoff=False) | 39.5% (79/200) | 36.0% (36/100) |

## 3. 完整實驗設定（experiment_config.yaml）

```yaml
schema_version: 1
environment:
  target_os: Windows-11
  python_minor: '3.11'
  torch_version: 2.5.1
  torch_cuda_index_url: https://download.pytorch.org/whl/cu124
  strict_env_check: false
  packages:
    transformers: 4.50.0
    trl: 0.16.0
    peft: 0.22.0
    accelerate: 1.4.0
    datasets: 3.6.0
    math-verify: 0.5.2
    latex2sympy2-extended: 1.0.6
    huggingface-hub: 0.36.2
    safetensors: 0.8.0
    tensorboard: 2.21.0
    pyyaml: 6.0.4
  env_vars:
    PYTHONPATH: src
    PYTHONUTF8: '1'
    TOKENIZERS_PARALLELISM: 'false'
hardware:
  cuda_visible_devices: null
  precision: auto
  attn_implementation: auto
  use_vllm: false
  gpu_profile: auto
  auto_adjust_training_hparams: false
  gpu_profiles:
    vram_6gb:
      eval_batch_size: 4
      gradient_checkpointing: true
      training_overrides:
        num_generations: 4
        per_device_train_batch_size: 4
        max_completion_length: 160
    vram_8gb:
      eval_batch_size: 16
      gradient_checkpointing: true
      training_overrides:
        num_generations: 8
        per_device_train_batch_size: 8
        max_completion_length: 192
    vram_12gb:
      eval_batch_size: 24
      gradient_checkpointing: true
      training_overrides:
        num_generations: 8
        per_device_train_batch_size: 8
        max_completion_length: 256
    vram_16gb:
      eval_batch_size: 32
      gradient_checkpointing: true
      training_overrides:
        num_generations: 8
        per_device_train_batch_size: 8
        max_completion_length: 256
    vram_24gb_plus:
      eval_batch_size: 32
      gradient_checkpointing: true
      training_overrides:
        num_generations: 8
        per_device_train_batch_size: 8
        max_completion_length: 256
    mps_apple_silicon:
      eval_batch_size: 4
      gradient_checkpointing: true
      training_overrides:
        num_generations: 4
        per_device_train_batch_size: 4
        max_completion_length: 192
model:
  name_or_path: models/Qwen2.5-3B-Instruct
  hf_repo_id: Qwen/Qwen2.5-3B-Instruct
  revision: main
  trust_remote_code: false
  lora:
    enabled: true
    r: 8
    alpha: 16
    dropout: 0.05
    target_modules:
    - q_proj
    - v_proj
  quantization:
    load_in_4bit: false
    bnb_4bit_quant_type: nf4
    use_bnb_nested_quant: false
dataset:
  name: datas/math_200
  train_file: datas/math_200/train.jsonl
  eval_file: datas/eval_100.jsonl
  auto_generate: true
  train_per_cat: 20
  eval_per_cat: 10
  seed: 2026
  system_prompt: You are a helpful math assistant. Solve the problem step by step
    and put the final answer inside \boxed{}.
training:
  seed: 42
  max_steps: 200
  num_train_epochs: 3
  num_generations: 4
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 2
  gradient_checkpointing: true
  gradient_checkpointing_use_reentrant: false
  learning_rate: 5.0e-05
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  max_prompt_length: 256
  max_completion_length: 192
  logging_steps: 1
  logging_first_step: true
  logging_strategy: steps
  log_completions: false
  log_level: info
  save_strategy: steps
  save_steps: 50
  save_total_limit: 2
  report_to:
  - tensorboard
  push_to_hub: false
  do_eval: false
  eval_strategy: 'no'
  overwrite_output_dir: true
gpg:
  reward_funcs:
  - accuracy
  reward_weights:
  - 1.0
  scale_rewards: false
  min_inverse_alpha: 0.0
  weighted_sample: true
  sample_strategy: uniform
  cooldown_steps: 20
  cooldown_backoff: false
  cooldown_backoff_factor: 2.0
  cooldown_max_steps: 0
  scale_batch: false
  adjust_lr: false
  adjust_gd: false
evaluation:
  do_sample: false
  max_new_tokens: 256
  batch_size: 8
experiments:
  run_tag: exp1_qwen3b
  output_dir_template: outputs/exp{run_tag}_{slug}
  report_md: outputs/comparison_report_{run_tag}.md
  report_json: outputs/comparison_report_{run_tag}.json
  groups:
  - slug: baseline_gpg
    label: GPG 原版 (Uniform)
    enabled: true
    overrides:
      weighted_sample: false
  - slug: uniform_cooldown
    label: GPG + Cooldown ({sample_strategy}, cd={cooldown_steps}, backoff={cooldown_backoff})
    enabled: true
    overrides:
      weighted_sample: true
      sample_strategy: '{sample_strategy}'
      cooldown_steps: '{cooldown_steps}'
      cooldown_backoff: '{cooldown_backoff}'
      cooldown_backoff_factor: '{cooldown_backoff_factor}'
      cooldown_max_steps: '{cooldown_max_steps}'
  - slug: safesoft_cooldown
    label: GPG + Safe-Soft Cooldown (Ours-Soft, cd={cooldown_steps}, backoff={cooldown_backoff})
    enabled: false
    overrides:
      weighted_sample: true
      sample_strategy: soft
      cooldown_steps: '{cooldown_steps}'
      cooldown_backoff: '{cooldown_backoff}'
      cooldown_backoff_factor: '{cooldown_backoff_factor}'
      cooldown_max_steps: '{cooldown_max_steps}'
```
