# GPG + Cooldown 團隊開發與訓練指南

本文件說明本分支相較於官方 [`AMAP-ML/GPG`](https://github.com/AMAP-ML/GPG) 的程式改動、Windows / RTX 4060 8GB 環境建置、訓練與對照實驗啟動方式，以及超參數調整細節。

---

## 1. 專案概述與核心改動

![GPG vs Cooldown Architecture](open-r1/assets/gpg_vs_cooldown_architecture.png)

針對本地單卡 8GB VRAM 環境與動態題目抽樣機制，主要修改了 `open-r1/src/open_r1/` 底下的三個檔案並新增本地實驗腳本：

1. **`open-r1/src/open_r1/configs.py`**
   - 於 `GPGConfig` 新增 `cooldown_steps: int = 20`（某題在群組 rollouts 中全對後，冷卻多少個 `global_step` 才重新加入抽樣池複習）。
   - `sample_strategy` 的說明補齊支援選項：`hard`、`uniform`、`soft`、`medium`。

2. **`open-r1/src/open_r1/gpg_trainer.py`**
   - **逐 batch 動態抽樣與冷卻機制（`AdpativeRepeatRandomSampler`）**：從第 1 個 epoch 起即改為逐 batch 依 `global_step` 動態計算抽樣權重（`rng = random.Random(base_seed + global_step)`）。在 `compute_loss` / reward 記錄階段，若題目平均分數 `score >= 1.0`（該題 rollouts 全對），將 `next_eligible_step[p]` 設為 `current_step + cooldown_steps`；未全對則重設為 `0` 立即恢復抽樣。若所有題目皆處於冷卻或權重為 0，會自動退回均勻抽樣防止 `random.choices` 報錯。
   - **修正 `prompt` 型別判斷 bug**：`make_conversation` 產生的 `prompt` 為 `list[dict]`，原版誤判為 `(dict,)`，已修正為 `isinstance(..., (list,))`。
   - **改良 `weight_soft`（Safe-Soft）**：僅取最近 3 次紀錄（`rewards[-3:]`）計算滑動平均，且對全錯或低分題保留 `0.2` 保底權重（`max(0.2, 1.0 - mean)`），避免難題被永久封殺導致題型偏食。
   - **生成階段啟用 KV Cache**：於 `unwrapped_model.generate(...)` 期間暫時切換至 `.eval()` 並設定 `use_cache=True`，避免 `gradient_checkpointing` 鎖死 `use_cache=False` 造成 $O(L^2)$ 重複計算，生成結束後再透過 `finally` 還原狀態。
   - **Checkpoint 額外狀態保存**：儲存模型 checkpoint 時，同步寫出 `next_eligible_step.json`（與 `data_weight.json` 同層），方便檢查冷卻狀態或續訓。

3. **`open-r1/src/open_r1/rewards.py`**
   - **Windows `math_verify` Patch**：Windows 沒有 `signal.SIGALRM`，`math_verify` 預設每次 `parse`/`verify` 會 spawn 子進程導致 `WinError 6` 且驗算恆為 `0`。在非 POSIX 系統上自動將 `math_verify` 的 `timeout` 替換為 no-op，確保數學答案批改正常運作。

4. **新增實驗與輔助檔案**
   - `open-r1/experiment_config.yaml`：**唯一集中設定檔（Single Source of Truth）**，統一管理 Windows 11 套件版本、CUDA GPU 自動相容規則、模型/LoRA/量化、資料集、訓練排程、GPG Cooldown 與對照實驗組別。
   - `open-r1/src/open_r1/experiment_config.py`：設定解析、Windows 11 套件檢查與不同 CUDA GPU 自動偵測/適配模組。
   - `open-r1/scripts/setup_win11.ps1`：Windows 11 一鍵建置 `openr1_venv` 與套件同步腳本。
   - `open-r1/recipes/local_rtx4060_gpg_cooldown.yaml`：單卡 RTX 4060 8GB 舊版平坦設定檔（仍保留向後相容）。
   - `open-r1/scripts/compare_gpg_vs_cooldown.py`：由 `experiment_config.yaml` 驅動的一鍵對照實驗腳本（支援 `--dry_run`、`--auto_download`、`--model_name_or_path`、`--run_tag`），並於報表自動寫入硬體與套件環境指紋。
   - `open-r1/scripts/generate_math_dataset.py`：確定性數學題庫生成器（10 大題型、嚴格零重疊，Train 200 題 + Eval 100 題），已預先生成並附在 `open-r1/datas/math_200/train.jsonl` 與 `open-r1/datas/eval_100.jsonl`。

---

## 2. 環境建置（Windows 11 / 跨 CUDA GPU 實測版本）

本機實際測試通過的套件版本集中鎖定在 **`open-r1/experiment_config.yaml`** 的 `environment` 區段（`openr1_venv/` 與 `models/` 已被 `.gitignore` 排除，clone 後需建立）：

### 方法 A（推薦）：Windows 11 一鍵建置與驗證

```powershell
cd open-r1
powershell -ExecutionPolicy Bypass -File scripts\setup_win11.ps1
```

腳本會自動檢查 Python 3.11、建立 `openr1_venv`、依 `experiment_config.yaml` 安裝鎖定版本的 CUDA PyTorch 與全部核心套件（不會執行 `pip install -e .`），並執行 `--check --strict` 驗證環境與 GPU。

> 若組員使用 **RTX 50 系列（Blackwell, CC 12.0）**，請先將 `experiment_config.yaml` 的 `torch_version` 與 `torch_cuda_index_url` 改為支援 CUDA 12.8 (`cu128`) 的 PyTorch 2.7+ 後再執行 `setup_win11.ps1`。

### 檢查現有環境是否與團隊一致

```powershell
cd open-r1
$env:PYTHONPATH = "src"
.\openr1_venv\Scripts\python.exe -m open_r1.experiment_config --check --strict
```

若只要印出修復指令而不中斷，可省略 `--strict`，或加上 `--print-setup` 印出完整 PowerShell 安裝指令。

### 跨 CUDA GPU 自動相容規則

`experiment_config.py` 在執行前會自動偵測本機顯示卡的 Compute Capability 與 VRAM：

| 偵測項目 | 自動適配行為 |
|---|---|
| **Precision (`auto`)** | Ampere / Ada / Blackwell (`CC >= 8.0`) 用 `bf16`；Turing / Volta (`CC 7.x`，如 RTX 2060 / GTX 1660) 自動降級為 `fp16`；無 CUDA 退回 `fp32`。 |
| **Attention (`auto`)** | Windows 下固定使用 PyTorch 內建 `sdpa`（Linux 且有 `flash_attn` + `CC >= 8.0` 才啟用 `flash_attention_2`）。 |
| **VRAM Profile (`auto`)** | 依顯存自動歸類為 `vram_6gb` / `vram_8gb` / `vram_12gb` / `vram_16gb` / `vram_24gb_plus`，自動調整評測 `eval.batch_size` 與 `gradient_checkpointing`。 |
| **公平對照鎖定** | `hardware.auto_adjust_training_hparams` 預設為 `false`，確保不同顯卡組員跑出的 `num_generations`、`max_completion_length`、`batch_size` 完全一致；若 VRAM 不足 6GB 想自動縮減生成參數，可於 YAML 改為 `true`。 |

### 方法 B：手動建置虛擬環境（PowerShell）

> **注意（Windows 請勿執行 `pip install -e .`）**：`open-r1/setup.py` 綁定了僅支援 Linux 的 `deepspeed==0.15.4`、`liger_kernel==0.5.3`、`vllm==0.7.2` 與 `bitsandbytes`，在 Windows 上直接 `pip install -e .` 會失敗。請依 `experiment_config.yaml` 鎖定版本手動安裝：

```powershell
cd open-r1
python -m venv openr1_venv
.\openr1_venv\Scripts\python.exe -m pip install --upgrade pip

# 安裝 CUDA 12.4 版 PyTorch
.\openr1_venv\Scripts\pip.exe install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 安裝訓練與評測所需核心套件（不裝 deepspeed / vllm / liger_kernel）
.\openr1_venv\Scripts\pip.exe install `
  transformers==4.50.0 `
  trl==0.16.0 `
  peft==0.22.0 `
  accelerate==1.4.0 `
  datasets==3.6.0 `
  math-verify==0.5.2 `
  latex2sympy2_extended==1.0.6 `
  huggingface-hub==0.36.2 `
  safetensors==0.8.0 `
  tensorboard==2.21.0 `
  pyyaml==6.0.4
```

### Step 2：下載基礎模型

`open-r1/experiment_config.yaml` 預設讀取本地路徑 `models/Qwen2.5-0.5B-Instruct`（也可於執行 `compare_gpg_vs_cooldown.py` 時加上 `--auto_download` 自動抓取）：

```powershell
.\openr1_venv\Scripts\huggingface-cli.exe download Qwen/Qwen2.5-0.5B-Instruct `
  --local-dir models/Qwen2.5-0.5B-Instruct
```

---

### 方法 C：Google Colab（L4 / A100 40G）

開啟 [`open-r1/notebooks/colab_gpg_cooldown.ipynb`](open-r1/notebooks/colab_gpg_cooldown.ipynb)，照順序執行即可。重點：

- 程式碼與 `outputs/` 放在 Google Drive，斷線後重跑同一指令會從最後一個 checkpoint 續訓（冷卻狀態 `next_eligible_step.json` 會一併還原）。
- 模型權重放 Colab 本機磁碟（`/content/models`），每個 session 重新下載。
- Colab 沿用內建 PyTorch，環境檢查出現 `MISMATCH` 警告是預期的。
- 大模型可用 4-bit QLoRA：`--load_in_4bit true`（例如 Qwen2.5-14B 在 A100 40G 上）。

`compare_gpg_vs_cooldown.py` 另新增以下參數（皆可搭配 `--dry_run` 先檢查）：

| 參數 | 用途 |
|---|---|
| `--load_in_4bit true/false` | 開關 4-bit 量化（QLoRA） |
| `--lora_r` / `--lora_alpha` | LoRA rank 與 alpha |
| `--lora_target_modules` | 逗號分隔的層名，或 `all-linear`（q/k/v/o/gate/up/down） |
| `--learning_rate` | 學習率 |
| `--eval_max_new_tokens` / `--eval_batch_size` | 評測生成長度與 batch size |
| `--eval_only` | 不訓練，只評測已存在的 adapter |
| `--report_suffix` | 報告檔名後綴，重評時不覆蓋原報告（例如 `_eval512`） |

---

## 3. 啟動訓練方式

所有訓練與腳本皆於 **`GPG/open-r1` 目錄** 執行（執行集中設定模組時會自動補齊 `PYTHONPATH` 與 `PYTHONUTF8=1`）：

### 方法 1（推薦）：集中設定檔 + 一鍵對照實驗

所有實驗參數皆在 `open-r1/experiment_config.yaml` 修改；要測試不同模型或超參數，只需修改 `model:` 與 `experiments.run_tag`（以免覆蓋舊報告），並先用 `--dry_run` 檢查實際解析結果：

```powershell
cd open-r1

# 1. 先做 dry-run，確認 GPU 偵測、套件版本、輸出路徑皆正確（不會啟動訓練）
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py --dry_run

# 2. 依 experiment_config.yaml 執行完整對照實驗
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py

# 3. 不改 YAML 臨時換模型測試（若本地沒有會自動從 --model_name_or_path 或 hf_repo_id 下載）
.\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py `
  --model_name_or_path models/Qwen2.5-1.5B-Instruct `
  --run_tag qwen15b_200 `
  --include_safesoft
```

> 若目標 `output_dir` 內已存在 `trainer_state.json`（且 `global_step >= max_steps`）與 `adapter_model.safetensors`，腳本會自動 `[SKIP]` 跳過訓練直接進入評測；若要重跑，請先刪除對應 `outputs/exp*` 資料夾或更改 `run_tag`。

### 方法 2：單獨啟動訓練（直接呼叫 `gpg.py`）

`gpg.py` 已支援直接傳入集中設定檔 `--config experiment_config.yaml`（會自動轉為相容的平坦 TRL YAML 並套用硬體適配），也維持相容舊版 `recipes/*.yaml`：

```powershell
$env:PYTHONPATH = "src"; $env:PYTHONUTF8 = "1"

# 使用集中設定檔啟動單組訓練
.\openr1_venv\Scripts\python.exe src/open_r1/gpg.py --config experiment_config.yaml

# 覆蓋單一參數啟動 Baseline
.\openr1_venv\Scripts\python.exe src/open_r1/gpg.py `
  --config experiment_config.yaml `
  --output_dir outputs/exp200_baseline_gpg `
  --weighted_sample False
```

### 方法 3：重新生成數學資料集（選用）

Repo 已內附 `open-r1/datas/math_200/train.jsonl`（200 題）與 `open-r1/datas/eval_100.jsonl`（100 題）。如需重新生成（`seed=2026`，10 種題型各 Train 20 / Eval 10 題，嚴格去重且零重疊）：

```powershell
.\openr1_venv\Scripts\python.exe scripts/generate_math_dataset.py
```

### 監控訓練曲線（TensorBoard）

```powershell
.\openr1_venv\Scripts\tensorboard.exe --logdir outputs
```

---

## 4. 超參數改動對照表

以下將目前實測設定 `open-r1/recipes/local_rtx4060_gpg_cooldown.yaml` 與 **官方 `GPGConfig` 預設值 / 官方 1.5B recipe（`recipes/Qwen2.5-1.5B-Instruct/gpg/config_v0.yaml`）** 做對照。

### A. GPG 演算法與動態抽樣參數

| 參數 | 官方預設 / `config_v0.yaml` | 本分支設定（`local_rtx4060_gpg_cooldown.yaml`） | 修改原因與說明 |
|---|---|---|---|
| `weighted_sample` | `False`（預設） | `true` | 啟用自適應抽樣權重與冷卻調度機制。 |
| `sample_strategy` | `uniform`（預設） | `uniform` | 冷卻控制主要由 `cooldown_steps` 負責；另可切換 `soft`（已修補保底權重 `0.2` 與近 3 次滑動平均）、`hard`、`medium`。 |
| `cooldown_steps` | *無（本分支新增）* | `20` | **核心新增參數**。題目全對後冷卻 20 個 `global_step` 不抽樣，期滿恢復權重 `1.0` 複習，避免浪費算力在已熟題並防止遺忘。 |
| `min_inverse_alpha` | `0.5`（預設） | `0.0` | `inverse_alpha` 下限。設為 `0.0` 以觀察原始有效梯度比例；`train.sh` 原版多卡實驗則設 `0.5`。 |
| `scale_rewards` | `True`（`GRPOConfig` 預設） | `false` | 關閉組內 reward 標準差縮放（對應 GPG 原論文／`train.sh` 去掉 std bias 的設定）。 |
| `reward_funcs` | `["accuracy", "format", "tag_count"]` / `config_v0`: `format_v2, cosine` | `["accuracy"]` | 本地 200 題數學實驗直接以 `\boxed{}` 數學正確性（`accuracy_reward`）作為單一純淨訊號。 |
| `reward_weights` | `config_v0`: `[1.0, 2.0]` | `[1.0]` | 配合單一 `accuracy` reward function。 |

### B. 模型與 8GB VRAM 適配參數

| 參數 | `config_v0.yaml`（伺服器多卡） | 本分支設定（RTX 4060 8GB） | 修改原因與說明 |
|---|---|---|---|
| `model_name_or_path` | `DeepSeek-R1-Distill-Qwen-1.5B` | `models/Qwen2.5-0.5B-Instruct` | 縮小基座模型至 0.5B，確保單卡 8GB VRAM 可做 G=8 rollout。 |
| `attn_implementation` | `flash_attention_2` | `sdpa` | Windows 無須額外編譯 FlashAttention-2，使用 PyTorch 內建 SDPA。 |
| `use_peft` | `false`（全參數微調） | `true` | 改用 LoRA 微調以大幅壓低優化器與梯度顯存需求。 |
| `lora_r` / `lora_alpha` | *未啟用* | `8` / `16` | 輕量 LoRA rank 設定（`lora_dropout: 0.05`）。 |
| `lora_target_modules` | *未啟用* | `["q_proj", "v_proj"]` | 僅針對 Attention 的 Q/V 投影層做 LoRA，兼顧顯存與訓練速度。 |
| `torch_dtype` / `bf16` | `bfloat16` / `true` | `bfloat16` / `true` | RTX 40 系列（Ada 架構）原生支援 bf16。 |
| `gradient_checkpointing` | `true`（`use_reentrant: false`） | `true`（`use_reentrant: false`） | 節省啟動值顯存；配合本分支在 `generate()` 暫時開啟 KV Cache 的修正，不影響推論速度。 |
| `use_vllm` | `false`（部分 7B recipe 為 `true`） | `false` | Windows 不支援 vLLM，直接使用 Transformers `generate`。 |

### C. 訓練排程與生成長度參數

| 參數 | 官方預設 / `config_v0.yaml` | 本分支設定（`local_rtx4060_gpg_cooldown.yaml`） | 修改原因與說明 |
|---|---|---|---|
| `learning_rate` | `1.0e-06` | `5.0e-05` | 因從全參數微調改為 LoRA（`r=8`），學習率需放大（約 $50\times$）才能有效收斂。 |
| `lr_scheduler_type` | 預設 `linear` / `config_v0`: `cosine_with_min_lr` | `cosine` | 搭配 `warmup_ratio: 0.1` 做餘弦退火。 |
| `warmup_ratio` | `0.0`（預設） / `config_v0`: `0.1` | `0.1` | 前 10% 步數線性暖身，穩定初期 RL 策略梯度。 |
| `num_generations` | `8`（預設） / `config_v0`: `6` | `8` | 每題採樣 8 條推理路徑（$G=8$），讓組內正負樣本對比與全對判斷更穩定。 |
| `per_device_train_batch_size` | `8`（預設） / `config_v0`: `6` | `8` | 每步生成 1 題 × 8 rollouts（因本設定檔下必須為 `num_generations` 的整數倍）。 |
| `gradient_accumulation_steps` | `1` | `1` | 單卡每步直接更新一次參數（對應 1 題 / step，200 steps 剛好掃過 200 題規模）。 |
| `max_prompt_length` | `512` | `256` | 本地數學題 prompt 較短，縮減為 256 節省顯存。 |
| `max_completion_length` | `256`（預設） / `config_v0`: `3584` | `192` | 配合 0.5B 模型與 8GB VRAM 限制單題生成長度；注意評測腳本 `evaluate_model` 使用 `max_new_tokens=256`。 |
| `max_steps` / `num_train_epochs` | `config_v0`: `500` / `1` | `200` / `3` | 以 `max_steps: 200` 控制總步數（200 步剛好掃過 200 題規模；先前 60 步實測單組約耗時 615–734 秒）。 |
| `save_strategy` / `save_steps` / `save_total_limit` | `steps` / `50` / 無限制 | `steps` / `50` / `2` | 每 50 步存一次並只保留最近 2 個 checkpoint，避免佔滿硬碟。 |
| `seed` | `42` | `42` | 固定隨機種子確保對照實驗可復現。 |

### 換到更大顯卡 / 更大資料集時的調整建議

若組員使用 RTX 3090 / 4090（24GB+）或 A100 跑更大題庫，建議優先調整以下四項：
1. `per_device_train_batch_size`：保持為 `num_generations` 的整數倍（例如設 `16` 或 `32`，代表每步取 2 或 4 題），並視需求拉大 `max_completion_length`（如 `512`–`1024`）。
2. `cooldown_steps`：建議依資料集大小與每步題數同步調整；當每步只抽 1 題時，`cooldown_steps=20` 約等於該題冷卻 20 題的間隔。
3. 若想做多輪對比實驗，優先使用 `--sample_strategy uniform` 與 `--sample_strategy soft` 搭配不同 `--cooldown_steps` 觀察 `inverse_alpha` 與泛化準確率變化。
4. 若從 0.5B 換回 1.5B / 7B 全參微調，記得將 `learning_rate` 降回 `1.0e-06` 左右。

---

## 5. 現有實驗結果與輸出位置

- **60 步初步對照報告**（已附在 repo 中）：見 [open-r1/outputs/comparison_report.md](open-r1/outputs/comparison_report.md)（含詳細數值 JSON：[comparison_report.json](open-r1/outputs/comparison_report.json)）。
  - 在 60 steps（Train-24 / Eval-20）預實驗中，GPG + Cooldown 的後 10 步平均 Reward 達 `0.6250`（原版 `0.2250`）、平均有效梯度比例 `inverse_alpha` 提升至 `0.408`（原版 `0.308`）、且 `60/60` 步全為非零梯度（原版 `57/60`），Train-24 留存率 `62.5%` 高於原版 `58.3%`。
- **200 步完整對照報告**：執行 `scripts/compare_gpg_vs_cooldown.py --max_steps 200` 後，會自動輸出至：
  - `open-r1/outputs/comparison_report_200.md`
  - `open-r1/outputs/comparison_report_200.json`
