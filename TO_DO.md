# GPG + Cooldown 團隊分工與單一操縱變因實驗清單

所有實驗統一由 `open-r1/experiment_config.yaml` 控制共同變因，**每組只動 1 個操縱變因**，並透過 `--run_tag` 隔離輸出目錄與報告檔名。

---

## 0. 共同控制變因（所有機器一致鎖定）

- **主錨點模型**（Exp 3–6）：`Qwen/Qwen2.5-1.5B-Instruct`
- **資料集**：`datas/math_200`（Train 200 / Eval 100，`seed=2026`）
- **訓練超參數**：`max_steps=200`、`G=4`、`per_device_train_batch_size=4`、`gradient_accumulation_steps=2`（= 2 題/step）、`L=192`、`learning_rate=5e-5`、`LoRA r=8, alpha=16 (q_proj, v_proj)`、`seed=42`、`min_inverse_alpha=0.0`
- **精度**：NVIDIA CUDA (RTX 4060 / 3060 / 3090) 自動走原生 `bf16`；MacBook Pro M1 自動走 MPS `float32`（**M1 耗時不與 CUDA 橫比**）。

### 組員前置作業（只需做一次）

- **Windows 11（4060 / 3060 / 3090）**：
  ```powershell
  cd open-r1
  powershell -ExecutionPolicy Bypass -File scripts\setup_win11.ps1
  ```
- **macOS Apple Silicon（MacBook Pro M1）**：
  ```bash
  cd open-r1
  bash scripts/setup_macos_m1.sh
  ```

---

## 核心比較模型一覽表（4 個模型）

| 模型名稱（HF Repo ID） | 本地路徑 | 參數量 | `bf16` 權重顯存 | 負責顯卡 / 機器 | `run_tag` |
|---|---|---:|---:|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | `models/Qwen2.5-0.5B-Instruct` | 0.5B | ~1.0 GB | RTX 4060 Laptop 8G / MacBook Pro M1 | `exp1_qwen05b_g4` |
| `Qwen/Qwen2.5-1.5B-Instruct` | `models/Qwen2.5-1.5B-Instruct` | 1.5B | ~3.1 GB | RTX 4060 Laptop 8G（主錨點） | `exp1_qwen15b` |
| `Qwen/Qwen2.5-3B-Instruct` | `models/Qwen2.5-3B-Instruct` | 3B | ~6.2 GB | RTX 3060 12G | `exp1_qwen3b` |
| `Qwen/Qwen2.5-7B-Instruct` | `models/Qwen2.5-7B-Instruct` | 7B | ~15.2 GB | RTX 3090 24G | `exp1_qwen7b` |

---

## 1. 實驗矩陣與 4 台機器分工總覽

| 實驗 | 操縱變因（只動這項） | 測試水準 | 指派機器 | 實驗目的 |
|---|---|---|---|---|
| **Exp 1：模型規模** | `model.name_or_path` | `Qwen2.5-0.5B` / **`1.5B`** / `3B` / `7B` | 4060 (`0.5B`, `1.5B`)<br>3060 (`3B`)<br>3090 (`7B`) | 驗證模型變強、全對題增加後，Cooldown 優勢是否放大 |
| **Exp 2：跨家族泛化** | `model.name_or_path`（非 Qwen） | `SmolLM2-1.7B-Instruct`<br>`Llama-3.2-3B-Instruct`（需 HF token） | 4060 (`SmolLM2-1.7B`)<br>3060 (`Llama-3.2-3B`) | 驗證 GPG+Cooldown 不是只對 Qwen 有效 |
| **Exp 3：冷卻步數** | `cooldown_steps`（固定 1.5B） | `10` / `20` / `40` / `9999`（永不複習） | **RTX 3090 24G** | 找冷卻長度甜蜜點，並觀察不複習是否遺忘 |
| **Exp 4：Cooldown 指數退避** | `cooldown_backoff`（有無退避） | `false`（固定） vs `true (factor=2.0)`，於 `cd=10` 與 `cd=20` 各比一次 | **RTX 3090 24G**（1.5B）<br>**MacBook Pro M1**（0.5B 輕量對照） | 驗證連續全對後冷卻時間倍增（10→20→40）是否比固定冷卻更省算力且不遺忘 |
| **Exp 5：抽樣策略** | `sample_strategy`（固定 1.5B, `cd=20`） | `uniform` / `soft` / `hard` / `medium` | **RTX 3060 12G** | 比較 Safe-Soft（保底 0.2）與 Hard/Medium 搭配 Cooldown 的效果 |
| **Exp 6：採樣數與長度** | `num_generations` / `max_completion_length` | `G=8 (L=192)` 與 `L=384 (G=4)` | **RTX 3090 24G** | 檢驗 `G=4` 是否誤判全對、`L=192` 是否截斷推導 |
| **Exp 7：多 Seed 穩定性** | `seed`（0.5B 輕量重現性） | `seed=123` / `seed=2026` | **MacBook Pro M1** | 補多組隨機種子，確認改善非單一 seed 偶發波動 |

---

## 2. 各機器可直接貼上執行的指令（於 `open-r1` 目錄）

> 💡 同一組基準模型（例如 1.5B）只要其中一組跑過 `baseline_gpg`，後面消融組可加 `--only_cooldown`（若連 Base 評測也想省可加 `--skip_base_eval`）直接省一半時間！

### A. RTX 4060 Laptop 8G（Windows 11）

```powershell
# Exp 1a: Qwen2.5-0.5B-Instruct (G=4 公平基準，含 Baseline + Cooldown)
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-0.5B-Instruct `
  --hf_repo_id Qwen/Qwen2.5-0.5B-Instruct `
  --run_tag exp1_qwen05b_g4 --auto_download

# Exp 1b: Qwen2.5-1.5B-Instruct (主錨點 Baseline + Uniform Cooldown cd=20)
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --hf_repo_id Qwen/Qwen2.5-1.5B-Instruct `
  --run_tag exp1_qwen15b --auto_download

# Exp 2a: 跨家族 SmolLM2-1.7B-Instruct (免 HF token)
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/SmolLM2-1.7B-Instruct `
  --hf_repo_id HuggingFaceTB/SmolLM2-1.7B-Instruct `
  --run_tag exp2_smollm2_17b --auto_download
```

### B. RTX 3060 12G（Windows 11）

```powershell
# Exp 1c: Qwen2.5-3B-Instruct
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-3B-Instruct `
  --hf_repo_id Qwen/Qwen2.5-3B-Instruct `
  --run_tag exp1_qwen3b --auto_download

# Exp 2b: Llama-3.2-3B-Instruct (需先 huggingface-cli login 且通過授權)
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Llama-3.2-3B-Instruct `
  --hf_repo_id meta-llama/Llama-3.2-3B-Instruct `
  --run_tag exp2_llama32_3b --auto_download

# Exp 5: 抽樣策略消融 (固定 1.5B, cd=20, 可加 --only_cooldown 省時間)
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --hf_repo_id Qwen/Qwen2.5-1.5B-Instruct `
  --sample_strategy soft --only_cooldown --run_tag exp5_soft --auto_download

.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --sample_strategy hard --only_cooldown --skip_base_eval --run_tag exp5_hard

.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --sample_strategy medium --only_cooldown --skip_base_eval --run_tag exp5_medium
```

### C. RTX 3090 24G（Windows 11）

```powershell
# Exp 1d: Qwen2.5-7B-Instruct (注意強制保持 gradient_checkpointing=true)
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-7B-Instruct `
  --hf_repo_id Qwen/Qwen2.5-7B-Instruct `
  --grad_ckpt true --run_tag exp1_qwen7b --auto_download

# Exp 3: 冷卻步數消融 (固定 1.5B, 無指數退避)
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --hf_repo_id Qwen/Qwen2.5-1.5B-Instruct `
  --cooldown_steps 10 --cooldown_backoff false --only_cooldown --run_tag exp3_cd10 --auto_download

.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --cooldown_steps 40 --cooldown_backoff false --only_cooldown --skip_base_eval --run_tag exp3_cd40

.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --cooldown_steps 9999 --cooldown_backoff false --only_cooldown --skip_base_eval --run_tag exp3_cd_inf

# Exp 4a: Cooldown 指數退避有無對照 (固定 1.5B, cd=10/20 對照上面固定冷卻組)
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --cooldown_steps 10 --cooldown_backoff true --cooldown_backoff_factor 2.0 `
  --only_cooldown --skip_base_eval --run_tag exp4_cd10_backoff

.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --cooldown_steps 20 --cooldown_backoff true --cooldown_backoff_factor 2.0 `
  --only_cooldown --skip_base_eval --run_tag exp4_cd20_backoff

# Exp 6: G=8 與 L=384 消融
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --num_generations 8 --per_device_train_batch_size 8 --gradient_accumulation_steps 1 `
  --run_tag exp6_g8

.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --max_completion_length 384 --only_cooldown --skip_base_eval --run_tag exp6_l384
```

### D. MacBook Pro M1（macOS Apple Silicon MPS）

```bash
# Exp 4b: 0.5B 輕量指數退避對照 (固定 vs 退避)
./openr1_venv/bin/python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-0.5B-Instruct \
  --hf_repo_id Qwen/Qwen2.5-0.5B-Instruct \
  --cooldown_steps 10 --cooldown_backoff true --cooldown_backoff_factor 2.0 \
  --only_cooldown --run_tag exp4_m1_05b_backoff --auto_download

# Exp 7: 0.5B 多 Seed 驗證
./openr1_venv/bin/python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-0.5B-Instruct \
  --seed 123 --run_tag exp7_m1_05b_seed123

./openr1_venv/bin/python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-0.5B-Instruct \
  --seed 2026 --run_tag exp7_m1_05b_seed2026
```

---

## 3. 收回結果並自動產生總表

每位組員只需把跑完的 **`open-r1/outputs/comparison_report_<run_tag>.json`**（與 `.md`）放進同一個 `open-r1/outputs/` 資料夾，執行：

```powershell
# Windows 11
.\openr1_venv\Scripts\python.exe scripts\aggregate_reports.py
```
```bash
# macOS M1
./openr1_venv/bin/python scripts/aggregate_reports.py
```

會自動檢查控制變因一致性，並在 `open-r1/outputs/` 產出：
- `summary_all_experiments.md`（Markdown 總比較表）
- `summary_all_experiments.csv`（UTF-8 BOM，Excel 直接開啟不亂碼）
