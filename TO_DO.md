# GPG + Cooldown 團隊分工與單一操縱變因實驗清單

所有實驗統一由 `open-r1/experiment_config.yaml` 控制共同變因，**每組只動 1 個操縱變因**，並透過 `--run_tag` 隔離輸出目錄與報告檔名。

---

## 0. 共同控制變因（所有機器一致鎖定）

- **主錨點模型**（Exp 3–6）：`Qwen/Qwen2.5-1.5B-Instruct`
- **資料集**：`datas/math_200`（Train 200 / Eval 100，`seed=2026`）
- **訓練超參數**：`max_steps=200`、`G=4`、`per_device_train_batch_size=4`、`gradient_accumulation_steps=2`（= 2 題/step）、`L=192`、`learning_rate=5e-5`、`LoRA r=8, alpha=16 (q_proj, v_proj)`、`seed=42`、`min_inverse_alpha=0.0`
- **精度**：NVIDIA CUDA (RTX 4060 / 3060 / 3090) 自動走原生 `bf16`；MacBook Pro M1 自動走 MPS `float32`（**M1 耗時不與 CUDA 橫比**）。

### 組員完整操作流程（照順序做）

1. **同步最新程式碼**：先 `git clone`（首次）或 `git pull` 拉取最新 `main`。
2. **一鍵建置與驗證環境（只需做一次）**：
   - **Windows 11（4060 / 3060 / 3090）**：
     ```powershell
     cd open-r1
     powershell -ExecutionPolicy Bypass -File scripts\setup_win11.ps1
     .\openr1_venv\Scripts\python.exe -m open_r1.experiment_config --check --strict
     ```
   - **macOS Apple Silicon（MacBook Pro M1）**：
     ```bash
     cd open-r1
     bash scripts/setup_macos_m1.sh
     ./openr1_venv/bin/python -m open_r1.experiment_config --check --strict
     ```
3. **開跑前先 Dry-Run 檢查**：執行對應實驗指令時先加上 `--dry_run`（確認 GPU 偵測、模型路徑與 `run_tag` 正確、不會開始訓練）。
4. **依所屬機器複製第 2 節指令開跑**：切勿自行修改控制變因（如 `learning_rate`、`lora.r`、`dataset.seed`）；遇到需要 `--auto_download` 的組別會自動抓模型，不要把 `models/` 或 `outputs/exp*/` checkpoint 推上 Git。
5. **將實驗報告推回 GitHub**（詳見第 3 節）：只推 `open-r1/outputs/comparison_report_<run_tag>.md` 與 `.json`。
6. **總彙整**：全員推完後 `git pull`，執行 `scripts/aggregate_reports.py` 產出總表並推回 repo。

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

## 3. 交回報告與自動產生總表

### Step A：組員將實驗報告推上 GitHub（於 repo 根目錄 `GPG/` 執行）

`open-r1/.gitignore` 已設定排除訓練 checkpoint（`outputs/exp*/`）與本地模型（`models/`），**只放行報告檔**，組員可直接把跑完的 `comparison_report_<run_tag>.*` 推上來：

```bash
git pull --rebase
git add open-r1/outputs/comparison_report_<run_tag>.md open-r1/outputs/comparison_report_<run_tag>.json
git commit -m "新增 <run_tag> 實驗報告"
git push
```

> ⚠️ **注意**：切勿強制 `git add -f models/` 或 `git add -f open-r1/outputs/exp*/`，單檔超過 100 MB 會造成整個 repo push 失敗。

### Step B：拉回所有報告並自動產生總表（於 `open-r1/` 目錄執行）

待組員都把報告 push 上 `main` 後，先執行 `git pull` 拉齊全部 `open-r1/outputs/comparison_report_*.json`，再執行彙整腳本：

```powershell
# Windows 11
git pull
.\openr1_venv\Scripts\python.exe scripts\aggregate_reports.py
```
```bash
# macOS M1
git pull
./openr1_venv/bin/python scripts/aggregate_reports.py
```

腳本會自動檢查各報告的控制變因一致性（並提示 M1 MPS 與 CUDA 耗時不可橫比），在 `open-r1/outputs/` 產出：
- `summary_all_experiments.md`（Markdown 總比較表）
- `summary_all_experiments.csv`（UTF-8 BOM，Excel 直接開啟不亂碼）

產生後即可一併推送總表存檔：

```bash
git add open-r1/outputs/summary_all_experiments.md open-r1/outputs/summary_all_experiments.csv
git commit -m "彙整全員實驗結果總表"
git push
```

---

## 4. 第二階段：Google Colab 實驗（L4 24G / A100 40G）

TO_DO 第 1–3 節的實驗大致完成後，第二階段改用 Google Colab。目的有兩個：一是確認第一階段的結論站得住（多 seed、評測不截斷），二是往更大的模型推（7B 診斷、14B 4-bit）。

### 4-0. 第二階段共同控制變因

- **沿用第 0 節的所有控制變因**（資料集、`max_steps=200`、`G=4`、`L=192`、`lr=5e-5`、`LoRA r=8 (q_proj, v_proj)`、`seed=42`），每組只動表格列出的操縱變因。
- **精度**：L4 與 A100 都是 CC ≥ 8.0，自動走 `bf16`；attention 走 `sdpa`。
- **4-bit 只用在 14B**：同一組比較（GPG 原版 vs Cooldown）內的 4-bit 與 LoRA 設定必須一致；4-bit 組不和第一階段的 bf16 組直接比準確率。
- **耗時只在同一張 GPU 內比較**：L4、A100 與第一階段的 4060 / 3060 / 3090 不橫比。
- **`run_tag` 一律用 `exp8_`～`exp11_` 開頭**，報告才能被 `aggregate_reports.py` 一起彙整。

### 4-1. Colab 操作流程（照順序做）

1. 在 Colab 開啟 `open-r1/notebooks/colab_gpg_cooldown.ipynb`，執行階段選 **L4** 或 **A100**（依下方分工）。
2. 依序執行 notebook 步驟 1–4：確認 GPU → 掛載 Drive 並 clone / pull repo（程式碼與 `outputs/` 放在 `MyDrive/GPG-Cooldown`）→ 安裝套件 → 設定環境變數（模型放 `/content/models`）。
3. **每台第一次使用時先跑步驟 5 的煙霧測試**（0.5B + 4-bit、5 步），確認整條流程可以跑通。
4. 執行下方 4-3 的指令。在 notebook cell 中，每行指令前面加 `!`，或在 cell 第一行寫 `%%bash`。**每個指令先加 `--dry_run` 檢查一次**。
5. **Colab 斷線時**：重新執行步驟 1–4，再**原指令重跑**。會從最後一個 checkpoint（每 50 步）續訓，冷卻狀態也會一併還原。
6. 跑完後依第 3 節推回報告（notebook 步驟 7 也可以直接推）。

> 💡 Repo 若是 private，要先在 Colab 左側 🔑 Secrets 新增 `GITHUB_TOKEN`，clone 與 push 都會用到。

### 4-2. 第二階段實驗矩陣與 Colab 分工

| 實驗 | 操縱變因（只動這項） | 測試水準 | 指派 GPU | 實驗目的 |
|---|---|---|---|---|
| **Exp 8：1.5B 多 Seed 主結論驗證** | `seed` | `42` / `123` / `2026`（Baseline + Cooldown 都跑） | **Colab L4** | 第一階段 1.5B「Cooldown 78% vs 原版 64%」只跑過 1 個 seed。用 3 個 seed 取平均，確認差距不是偶然 |
| **Exp 9：評測截斷檢查** | `evaluation.max_new_tokens` | 訓練 `L=384`，評測 `512`（必要時再用 `--eval_only` 重評 `768`） | **Colab L4** | Exp 6 L=384 訓練 reward 0.75，但 Eval 只有 46%，懷疑是評測只給 256 tokens、答案被截斷 |
| **Exp 10：7B 學不起來診斷** | LoRA 容量 + `learning_rate` | `r=16, alpha=32, all-linear, lr=1e-4` | **Colab A100 40G** | 第一階段 7B 只從 34% 升到 37%，兩組幾乎相同。確認是不是 LoRA 太小、學習率太低 |
| **Exp 11：14B 大模型（4-bit QLoRA）** | `model.name_or_path` + `load_in_4bit` | `Qwen2.5-14B-Instruct`, 4-bit NF4 | **Colab A100 40G** | 驗證更大模型上 Cooldown 是否仍有優勢；4-bit 權重約 9GB，40G 放得下 |

> 依賴關係：Exp 10 若顯示 7B 需要較大的 LoRA / 學習率才學得動，Exp 11 也改用相同的 LoRA 設定（指令中的選用版本）。

### 4-3. 各 GPU 可直接貼上執行的指令（於 `open-r1` 目錄，notebook 已自動 `cd`）

#### E. Colab L4 24G

```bash
# Exp 8: 1.5B 多 Seed（每個 seed 含 Baseline + Cooldown，約 1.5 小時 / seed）
python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-1.5B-Instruct \
  --hf_repo_id Qwen/Qwen2.5-1.5B-Instruct \
  --seed 42 --run_tag exp8_l4_qwen15b_seed42 --auto_download

python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-1.5B-Instruct \
  --seed 123 --skip_base_eval --run_tag exp8_l4_qwen15b_seed123

python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-1.5B-Instruct \
  --seed 2026 --skip_base_eval --run_tag exp8_l4_qwen15b_seed2026

# Exp 9: L=384 訓練 + 評測給 512 tokens（Baseline + Cooldown 都跑）
python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-1.5B-Instruct \
  --hf_repo_id Qwen/Qwen2.5-1.5B-Instruct \
  --max_completion_length 384 --eval_max_new_tokens 512 \
  --run_tag exp9_l4_l384_eval512 --auto_download

# Exp 9（選用）：不重訓，只用 768 tokens 重評同一批 adapter，報告另存為 *_eval768
python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-1.5B-Instruct \
  --max_completion_length 384 --run_tag exp9_l4_l384_eval512 \
  --eval_only --eval_max_new_tokens 768 --report_suffix _eval768
```

#### F. Colab A100 40G

```bash
# Exp 10: 7B 診斷（LoRA 掛全部線性層、r=16、lr=1e-4；Baseline + Cooldown）
python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-7B-Instruct \
  --hf_repo_id Qwen/Qwen2.5-7B-Instruct \
  --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --learning_rate 1e-4 \
  --grad_ckpt true --run_tag exp10_a100_qwen7b_r16_lr1e4 --auto_download

# Exp 11: 14B 4-bit QLoRA（沿用預設 LoRA r=8 q/v、lr=5e-5；Baseline + Cooldown）
python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-14B-Instruct \
  --hf_repo_id Qwen/Qwen2.5-14B-Instruct \
  --load_in_4bit true --grad_ckpt true \
  --run_tag exp11_a100_qwen14b_4bit --auto_download

# Exp 11（選用）：若 Exp 10 顯示需要較大的 LoRA，改用相同設定
python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-14B-Instruct \
  --load_in_4bit true --grad_ckpt true \
  --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --learning_rate 1e-4 \
  --run_tag exp11_a100_qwen14b_4bit_r16_lr1e4
```

> ⏱️ 14B 4-bit 生成速度比 bf16 慢，200 步可能要好幾個小時。時間不夠的話，兩組都加上 `--max_steps 100`（Baseline 和 Cooldown 一定要同步數），並把 `run_tag` 改成 `..._s100`。

### 4-4. 第二階段新增的指令參數

| 參數 | 用途 |
|---|---|
| `--load_in_4bit true/false` | 開關 4-bit 量化（QLoRA，需 `bitsandbytes`，Colab 已裝） |
| `--lora_r` / `--lora_alpha` | LoRA rank 與 alpha |
| `--lora_target_modules` | 逗號分隔的層名，或 `all-linear`（q/k/v/o/gate/up/down） |
| `--learning_rate` | 學習率 |
| `--eval_max_new_tokens` / `--eval_batch_size` | 評測生成長度與 batch size |
| `--eval_only` | 不訓練，只評測已存在的 adapter |
| `--report_suffix` | 報告檔名後綴，重評時不覆蓋原報告 |

> 報告的訓練表另外新增 **每步秒數** 與 **峰值顯存 (GB)** 兩欄，來源是訓練 log 裡的 `peak_vram_gb`。

### 4-5. 交回報告

和第 3 節相同，只推 `open-r1/outputs/comparison_report_exp8_*` ~ `exp11_*` 的 `.md` / `.json`。全部推完後重跑 `scripts/aggregate_reports.py` 更新總表。

---

## 5. 第三階段：極限性能實驗（Exp 12–13）

這個階段測兩種極限。**Exp 12** 測硬體：A100 40G / L4 最多塞得下多大的 G×L。**Exp 13** 測方法：把訓練拉長、設定疊滿後，GPG 原版和 Cooldown 各自的準確率天花板在哪裡。

### 5-0. 本階段控制變因

- **Exp 12 只看能不能跑與資源用量，不評測準確率**：每組只訓練 20 步，固定用 Cooldown 組設定。每步秒數取 Trainer 自己量的訓練時間，不含模型載入。
- **Exp 13 除了表格列出的操縱變因，其他沿用第 0 節**。Baseline 與 Cooldown 的步數、G、L、LoRA 必須相同。
- 耗時與顯存只在同一張 GPU 內比較。

### 5-1. 實驗矩陣與 Colab 分工

| 實驗 | 操縱變因（只動這項） | 測試水準 | 指派 GPU | 實驗目的 |
|---|---|---|---|---|
| **Exp 12a：7B 硬體極限** | `num_generations` × `max_completion_length` | G = `4/8/16` × L = `192/384/768`（共 9 組，各 20 步） | **Colab A100 40G** | 找出 7B（bf16）在 40G 上不 OOM 的最大 G×L，以及每步耗時怎麼隨 G、L 成長 |
| **Exp 12b：14B 4-bit 硬體極限** | 同上 | 同上 | **Colab A100 40G** | 找出 14B（4-bit）的可用範圍，決定 Exp 11 之後還能把 G、L 加到多大 |
| **Exp 12c：3B 在 L4 的硬體極限**（選用） | 同上 | 同上 | **Colab L4 24G** | 給 L4 一份可用設定表，之後排實驗才知道哪些組合要丟給 A100 |
| **Exp 13a：長訓練天花板** | `max_steps` | `1000` 步（G=8, L=192，對照 Exp 6 G=8 的 200 步） | **Colab L4 24G** | 看 Train-200 接近飽和時，Cooldown 是「更早到頂」還是「頂更高」。註：Exp 6 G=8 時原版 80% 勝過 Cooldown 75%，長訓練可確認這個差距會不會消失 |
| **Exp 13b：設定全開天花板** | `max_steps` + L + LoRA 容量 | `1000` 步、G=8、L=384（評測 512 tokens）、LoRA r=16 all-linear | **Colab A100 40G** | 把目前看起來有幫助的設定疊在一起，找出 1.5B 在這套題庫上的最高準確率 |

> 依賴關係：Exp 13b 的 G、L 若在 Exp 12 證明 A100 撐得住更大的組合，可以再開一組 `exp13c` 往上加，但 Baseline 與 Cooldown 一定要同設定。

### 5-2. 各 GPU 可直接貼上執行的指令（於 `open-r1` 目錄）

#### F. Colab A100 40G

```bash
# Exp 12a: 7B（bf16）G x L 掃描，產出 outputs/stress_report_exp12a_a100_qwen7b.md
python scripts/stress_test_limits.py \
  --model_name_or_path models/Qwen2.5-7B-Instruct \
  --hf_repo_id Qwen/Qwen2.5-7B-Instruct \
  --generations 4,8,16 --lengths 192,384,768 --steps 20 \
  --run_tag exp12a_a100_qwen7b --auto_download

# Exp 12b: 14B（4-bit）G x L 掃描
python scripts/stress_test_limits.py \
  --model_name_or_path models/Qwen2.5-14B-Instruct \
  --hf_repo_id Qwen/Qwen2.5-14B-Instruct \
  --load_in_4bit true \
  --generations 4,8,16 --lengths 192,384,768 --steps 20 \
  --run_tag exp12b_a100_qwen14b_4bit --auto_download

# Exp 13b: 設定全開天花板（1.5B, 1000 步, G=8, L=384, LoRA r=16 all-linear, 評測 512 tokens）
python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-1.5B-Instruct \
  --hf_repo_id Qwen/Qwen2.5-1.5B-Instruct \
  --max_steps 1000 \
  --num_generations 8 --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
  --max_completion_length 384 --eval_max_new_tokens 512 \
  --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear \
  --run_tag exp13b_a100_qwen15b_full_s1000 --auto_download
```

#### E. Colab L4 24G

```bash
# Exp 12c（選用）: 3B（bf16）G x L 掃描
python scripts/stress_test_limits.py \
  --model_name_or_path models/Qwen2.5-3B-Instruct \
  --hf_repo_id Qwen/Qwen2.5-3B-Instruct \
  --generations 4,8,16 --lengths 192,384,768 --steps 20 \
  --run_tag exp12c_l4_qwen3b --auto_download

# Exp 13a: 長訓練天花板（1.5B, 1000 步, G=8, L=192；Baseline + Cooldown）
python scripts/compare_gpg_vs_cooldown.py \
  --model_name_or_path models/Qwen2.5-1.5B-Instruct \
  --hf_repo_id Qwen/Qwen2.5-1.5B-Instruct \
  --max_steps 1000 \
  --num_generations 8 --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
  --run_tag exp13a_l4_qwen15b_g8_s1000 --auto_download
```

> ⏱️ 時間估計：Exp 6 G=8 在 3090 上 200 步約 32 分鐘 / 組，所以 Exp 13 的 1000 步約 2.5–3 小時 / 組（Baseline + Cooldown 共兩組）。Colab 斷線就原指令重跑，會從 checkpoint 續訓。Exp 12 每個組合幾分鐘，但 G=16、L=768 的組合可能 OOM，腳本會記下 `OOM` 並自動跳到下一組。

### 5-3. 新增腳本：`scripts/stress_test_limits.py`

| 參數 | 用途 |
|---|---|
| `--generations` | 要測的 G，逗號分隔（預設 `4,8,16`） |
| `--lengths` | 要測的 L，逗號分隔（預設 `192,384,768`） |
| `--steps` | 每組訓練步數（預設 20） |
| `--load_in_4bit` / `--lora_r` / `--lora_alpha` | 與 `compare_gpg_vs_cooldown.py` 相同 |
| `--cooldown false` | 改測 GPG 原版設定（預設測 Cooldown） |
| `--keep_outputs` | 保留每組 checkpoint（預設只留 log 與 trainer_state，省空間） |

輸出：`outputs/stress_report_<run_tag>.md` 與 `.json`，表格包含每組的狀態（OK / OOM / FAIL）、完成步數、每步秒數、峰值顯存（allocated / reserved）。

### 5-4. 交回報告

Exp 13 的報告和前面相同，推 `comparison_report_exp13*`。Exp 12 推 `open-r1/outputs/stress_report_exp12*.md` 與 `.json`，`.gitignore` 已放行。每組的 `run.log` 留在 Drive 就好，不用推。

