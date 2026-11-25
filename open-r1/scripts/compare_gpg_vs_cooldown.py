"""
統一對照實驗啟動器（由 experiment_config.yaml 集中驅動）：
比較：
1. Base Model（未訓練）
2. GPG 原版（Uniform 均勻抽樣）
3. GPG + Uniform Cooldown（Ours）
4. [可選 --include_safesoft 或 YAML enabled=true] GPG + Safe-Soft Cooldown

用法（於 C:\\Users\\chris\\Desktop\\GPG\\open-r1 執行）:
    .\\openr1_venv\\Scripts\\python.exe scripts/compare_gpg_vs_cooldown.py
    .\\openr1_venv\\Scripts\\python.exe scripts/compare_gpg_vs_cooldown.py --dry_run
    .\\openr1_venv\\Scripts\\python.exe scripts/compare_gpg_vs_cooldown.py --model_name_or_path models/Qwen2.5-1.5B-Instruct --run_tag qwen15b_200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for p in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_math_dataset import generate_datasets
from open_r1.experiment_config import (
    DEFAULT_CONFIG_PATH,
    build_env_snapshot,
    build_trl_config_dict,
    export_trl_yaml,
    load_and_resolve_config,
    print_status_report,
    resolve_experiment_groups,
)
from open_r1.rewards import accuracy_reward  # 內含 Windows math_verify patch


def _resolve_root_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p)


def ensure_datasets(resolved: dict[str, Any]) -> tuple[Path, Path]:
    """確認訓練與評測資料集存在且筆數符合設定，否則依設定自動生成。"""
    ds_cfg = resolved["dataset"]
    train_file = _resolve_root_path(ds_cfg["train_file"])
    eval_file = _resolve_root_path(ds_cfg["eval_file"])

    expected_train = int(ds_cfg.get("train_per_cat", 20)) * 10
    expected_eval = int(ds_cfg.get("eval_per_cat", 10)) * 10

    if train_file.exists() and eval_file.exists():
        with open(train_file, encoding="utf-8") as f:
            n_tr = sum(1 for line in f if line.strip())
        with open(eval_file, encoding="utf-8") as f:
            n_ev = sum(1 for line in f if line.strip())
        if n_tr == expected_train and n_ev == expected_eval:
            print(f"[DATA] 使用現有資料集：Train={n_tr} 題, Eval={n_ev} 題")
            return train_file, eval_file

    if not ds_cfg.get("auto_generate", True):
        raise FileNotFoundError(f"找不到資料集：{train_file} 或 {eval_file}")

    generate_datasets(
        train_per_cat=int(ds_cfg.get("train_per_cat", 20)),
        eval_per_cat=int(ds_cfg.get("eval_per_cat", 10)),
        seed=int(ds_cfg.get("seed", 2026)),
        train_file=train_file,
        eval_file=eval_file,
    )
    return train_file, eval_file


def ensure_model_available(resolved: dict[str, Any], auto_download: bool) -> Path | str:
    """確認基礎模型存在；若為不存在的本地路徑且開啟 auto_download，從 HuggingFace 下載。"""
    model_cfg = resolved["model"]
    raw_name = str(model_cfg["name_or_path"])
    local_candidate = _resolve_root_path(raw_name)
    if local_candidate.exists():
        return local_candidate

    # 若看起來是本地目錄路徑（含 models/ 或磁碟路徑）但不存在
    looks_local = raw_name.startswith(("models/", "models\\", "./", ".\\")) or Path(raw_name).is_absolute()
    if looks_local:
        hf_repo = model_cfg.get("hf_repo_id")
        if auto_download and hf_repo:
            from huggingface_hub import snapshot_download

            print(f"[MODEL] 本地模型不存在，開始下載 {hf_repo} -> {local_candidate} ...")
            local_candidate.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=str(hf_repo),
                revision=str(model_cfg.get("revision", "main")),
                local_dir=str(local_candidate),
            )
            return local_candidate
        raise FileNotFoundError(
            f"找不到本地模型路徑: {local_candidate}\n"
            f"請先下載模型（或加上 --auto_download 由 {hf_repo} 自動下載）。"
        )

    # 否則視為 HuggingFace Hub repo id
    return raw_name


def run_training_if_needed(
    name: str,
    output_dir: Path,
    trl_dict: dict[str, Any],
    env_snapshot: dict[str, Any],
    max_steps: int,
) -> None:
    """若該組實驗尚未跑完指定 steps，將解析後設定存檔並呼叫 gpg.py 執行。"""
    state_file = output_dir / "trainer_state.json"
    if state_file.exists():
        with open(state_file, encoding="utf-8") as f:
            st = json.load(f)
        if st.get("global_step", 0) >= max_steps and (output_dir / "adapter_model.safetensors").exists():
            print(f"[SKIP] {name} 已完成 ({st['global_step']} steps): {output_dir}")
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml_path = output_dir / "resolved_trl_config.yaml"
    snapshot_path = output_dir / "env_snapshot.json"
    export_trl_yaml(trl_dict, resolved_yaml_path)
    snapshot_path.write_text(json.dumps(env_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[RUN] 啟動 {name} 訓練（{max_steps} steps）...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    env["PYTHONUTF8"] = "1"
    cmd = [
        sys.executable,
        str(ROOT / "src" / "open_r1" / "gpg.py"),
        "--config",
        str(resolved_yaml_path),
    ]
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    print(f"[DONE] {name} 訓練完成！")


def parse_train_logs(output_dir: Path) -> dict[str, Any]:
    with open(output_dir / "trainer_state.json", encoding="utf-8") as f:
        st = json.load(f)
    with open(output_dir / "train_results.json", encoding="utf-8") as f:
        res = json.load(f)

    logs = [x for x in st["log_history"] if "reward" in x]
    rewards = [x["reward"] for x in logs]
    easy = [x.get("num_easy_problem", 0.0) for x in logs]
    hard = [x.get("num_hard_problem", 0.0) for x in logs]
    inv_a = [x.get("inverse_alpha", 0.0) for x in logs]
    grad_nz = sum(1 for x in logs if x.get("grad_norm", 0.0) > 0)
    peak_vram = max((x.get("peak_vram_gb", 0.0) for x in logs), default=0.0)
    n = len(rewards)
    q = max(1, n // 4)

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "steps": n,
        "runtime_sec": round(res.get("train_runtime", 0.0), 1),
        "avg_reward": round(avg(rewards), 4),
        "first_quarter_reward": round(avg(rewards[:q]), 4),
        "last_quarter_reward": round(avg(rewards[-q:]), 4),
        "avg_easy_per_step": round(avg(easy), 3),
        "avg_hard_per_step": round(avg(hard), 3),
        "avg_valid_grad_ratio": round(avg(inv_a), 3),
        "nonzero_grad_steps": f"{grad_nz}/{n}",
        "peak_vram_gb": round(peak_vram, 2),
        "sec_per_step": round(res.get("train_runtime", 0.0) / n, 2) if n else 0.0,
        "reward_curve": rewards,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _format_chat_prompt(tokenizer: PreTrainedTokenizerBase, system_prompt: str | None, problem: str) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": problem})
    try:
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    except Exception:
        # 部分模型不支援 system role，改將 system prompt 併入 user prompt
        merged = f"{system_prompt}\n\n{problem}" if system_prompt else problem
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": merged}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )


@torch.inference_mode()
def evaluate_model(
    name: str,
    model_source: Path | str,
    adapter_dir: Path | None,
    dataset: list[dict[str, Any]],
    resolved: dict[str, Any],
) -> dict[str, Any]:
    """依集中設定檔解析後的精度、attention 與解碼參數評測模型準確率。"""
    model_cfg = resolved["model"]
    hw_res = resolved["hardware_resolved"]
    eval_cfg = resolved["evaluation"]
    system_prompt = resolved["dataset"].get("system_prompt")
    batch_size = int(eval_cfg["batch_size"])
    max_new_tokens = int(eval_cfg["max_new_tokens"])
    do_sample = bool(eval_cfg["do_sample"])
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    dtype_name = hw_res["torch_dtype"]
    torch_dtype = getattr(torch, dtype_name)

    print(f"[EVAL] 評測 {name}（共 {len(dataset)} 題, dtype={dtype_name}, bs={batch_size}）...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_source), trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hw_det = resolved["hardware_detected"]
    backend = hw_det.get("accelerator_backend") or ("cuda" if torch.cuda.is_available() else "cpu")

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "attn_implementation": hw_res["attn_implementation"],
        "trust_remote_code": trust_remote_code,
    }
    if backend == "cuda":
        load_kwargs["device_map"] = "auto"

    quant_cfg = model_cfg.get("quantization", {}) or {}
    if quant_cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quant_cfg.get("use_bnb_nested_quant", False)),
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(str(model_source), **load_kwargs)
    if backend == "mps":
        model = model.to("mps")
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    prompts_text = [_format_chat_prompt(tokenizer, system_prompt, row["problem"]) for row in dataset]
    solutions = [row["solution"] for row in dataset]

    completions = []
    for i in range(0, len(prompts_text), batch_size):
        batch_prompts = prompts_text[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_ids = out[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for t in texts:
            completions.append([{"content": t}])

    scores = accuracy_reward(completions, solutions)
    correct = int(sum(scores))
    acc = correct / len(scores)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch, "mps", None) is not None and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return {
        "correct": correct,
        "total": len(scores),
        "accuracy": round(acc, 4),
        "per_item": scores,
    }


def render_comparison_report_md(
    resolved: dict[str, Any],
    train_count: int,
    eval_count: int,
    train_logs: dict[str, Any],
    eval_results: dict[str, Any],
) -> str:
    """組裝包含完整環境、關鍵超參數、比較結果與完整 experiment_config.yaml 的 Markdown 報告。"""
    hw = resolved["hardware_detected"]
    hr = resolved["hardware_resolved"]
    model_cfg = resolved["model"]
    lora = model_cfg["lora"]
    quant = model_cfg["quantization"]
    tr = resolved["training"]
    gpg = resolved["gpg"]
    ev = resolved["evaluation"]
    model_display = Path(str(model_cfg["name_or_path"])).name
    decode_mode = "Sampling" if ev["do_sample"] else "Greedy Decoding"
    max_steps = int(tr["max_steps"])
    cuda_ver = hw.get("cuda_version") or "N/A"
    backend = hw.get("accelerator_backend") or ("cuda" if hw.get("cuda_available") else ("mps" if hw.get("mps_available") else "cpu"))
    effective_yaml = yaml.safe_dump(resolved["effective_config"], allow_unicode=True, sort_keys=False).rstrip()

    lines = [
        f"# GPG 原版 vs. 改良版 對照報告（Train {train_count} / Eval {eval_count} / G={tr['num_generations']} / {max_steps} Steps）",
        "",
        f"1. **基礎模型**：`{model_display}`（`name_or_path={model_cfg['name_or_path']}`, `hf_repo_id={model_cfg.get('hf_repo_id')}` | `dtype={hr['torch_dtype']}` (`bf16={hr['bf16']}`, `fp16={hr['fp16']}`), `attn={hr['attn_implementation']}` | LoRA: `enabled={lora['enabled']}`, `r={lora['r']}`, `alpha={lora['alpha']}`, `dropout={lora['dropout']}`, `target_modules={lora['target_modules']}` | 4-bit: `load_in_4bit={quant['load_in_4bit']}`）",
        f"2. **硬體與環境**：GPU=`{hw['gpu_name']}` | backend=`{backend}` | VRAM=`{hw['total_vram_gb']} GB` (free `{hw['free_vram_gb']} GB`), `CC={hw['compute_capability']}`, `profile={hr['profile']}` | OS=`{hw['os']} {hw['os_release']}` | Python=`{hw['python_version']}` | PyTorch=`{hw['torch_version']}` (CUDA `{cuda_ver}`) | Config SHA256=`{resolved['config_sha256'][:12]}`",
        f"3. **評測方式**：{decode_mode}（`do_sample={ev['do_sample']}`, `max_new_tokens={ev['max_new_tokens']}`, `batch_size={ev['batch_size']}`）",
        f"4. **重要參數**：`Train={train_count}` / `Eval={eval_count}` / `G={tr['num_generations']}` / `Steps={max_steps}` | `L={tr['max_completion_length']}`, `max_prompt_length={tr['max_prompt_length']}`, `per_device_train_batch_size={tr['per_device_train_batch_size']}`, `gradient_accumulation_steps={tr['gradient_accumulation_steps']}`, `learning_rate={tr['learning_rate']}`, `lr_scheduler_type={tr['lr_scheduler_type']}`, `warmup_ratio={tr['warmup_ratio']}`, `cooldown_steps={gpg['cooldown_steps']}`, `cooldown_backoff={gpg.get('cooldown_backoff', False)}`, `cooldown_backoff_factor={gpg.get('cooldown_backoff_factor', 2.0)}`, `cooldown_max_steps={gpg.get('cooldown_max_steps', 0)}`, `sample_strategy={gpg['sample_strategy']}`, `scale_rewards={gpg['scale_rewards']}`, `min_inverse_alpha={gpg['min_inverse_alpha']}`, `seed={tr['seed']}`",
        "",
        "## 1. 訓練過程指標比較",
        "",
        "| 組別 | 耗時 (s) | 全程平均 Reward | 前 1/4 Reward | 後 1/4 Reward | 平均全對題/步 (`easy`) | 有效梯度比 (`inv_alpha`) | 非零梯度步數 | 每步秒數 | 峰值顯存 (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, log in train_logs.items():
        peak = log.get("peak_vram_gb") or "-"
        lines.append(
            f"| {name} | {log['runtime_sec']} | {log['avg_reward']:.4f} | {log['first_quarter_reward']:.4f} "
            f"| {log['last_quarter_reward']:.4f} | {log['avg_easy_per_step']:.3f} | {log['avg_valid_grad_ratio']:.3f} "
            f"| {log['nonzero_grad_steps']} | {log.get('sec_per_step', '-')} | {peak} |"
        )

    lines += [
        "",
        f"## 2. 確定性評測結果（Train-{train_count} 留存率 & Eval-{eval_count} 泛化力）",
        "",
        f"| 模型版本 | Train-{train_count} 留存率（答對/總數） | Eval-{eval_count} 泛化準確率（答對/總數） |",
        "|---|---:|---:|",
    ]
    for label, res in eval_results.items():
        tr_res = res["train_200"]
        ev_res = res["eval_100"]
        lines.append(
            f"| {label} | {tr_res['accuracy']*100:.1f}% ({tr_res['correct']}/{tr_res['total']}) "
            f"| {ev_res['accuracy']*100:.1f}% ({ev_res['correct']}/{ev_res['total']}) |"
        )

    lines += [
        "",
        "## 3. 完整實驗設定（experiment_config.yaml）",
        "",
        "```yaml",
        effective_yaml,
        "```",
    ]
    return "\n".join(lines) + "\n"


# Qwen2 / Qwen3 / Llama 家族的全部線性層（--lora_target_modules all-linear 時展開）
ALL_LINEAR_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _parse_bool_flag(raw: str) -> bool:
    low = raw.strip().lower()
    if low in ("true", "1", "yes", "y"):
        return True
    if low in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"無效的布林值: {raw!r}（請輸入 true 或 false）")


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.model_name_or_path:
        overrides.setdefault("model", {})["name_or_path"] = args.model_name_or_path
    if args.hf_repo_id:
        overrides.setdefault("model", {})["hf_repo_id"] = args.hf_repo_id
    if args.run_tag:
        overrides.setdefault("experiments", {})["run_tag"] = args.run_tag
    if args.max_steps is not None:
        overrides.setdefault("training", {})["max_steps"] = args.max_steps
    if args.num_generations is not None:
        overrides.setdefault("training", {})["num_generations"] = args.num_generations
        if args.per_device_train_batch_size is None:
            overrides.setdefault("training", {})["per_device_train_batch_size"] = args.num_generations
    if args.per_device_train_batch_size is not None:
        overrides.setdefault("training", {})["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        overrides.setdefault("training", {})["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.max_completion_length is not None:
        overrides.setdefault("training", {})["max_completion_length"] = args.max_completion_length
    if args.seed is not None:
        overrides.setdefault("training", {})["seed"] = args.seed
    if args.grad_ckpt is not None:
        overrides.setdefault("training", {})["gradient_checkpointing"] = args.grad_ckpt
    if args.sample_strategy:
        overrides.setdefault("gpg", {})["sample_strategy"] = args.sample_strategy
    if args.cooldown_steps is not None:
        overrides.setdefault("gpg", {})["cooldown_steps"] = args.cooldown_steps
    if args.cooldown_backoff is not None:
        overrides.setdefault("gpg", {})["cooldown_backoff"] = args.cooldown_backoff
    if args.cooldown_backoff_factor is not None:
        overrides.setdefault("gpg", {})["cooldown_backoff_factor"] = args.cooldown_backoff_factor
    if args.cooldown_max_steps is not None:
        overrides.setdefault("gpg", {})["cooldown_max_steps"] = args.cooldown_max_steps
    if args.precision:
        overrides.setdefault("hardware", {})["precision"] = args.precision
    if args.gpu_profile:
        overrides.setdefault("hardware", {})["gpu_profile"] = args.gpu_profile
    if args.load_in_4bit is not None:
        overrides.setdefault("model", {}).setdefault("quantization", {})["load_in_4bit"] = args.load_in_4bit
    if args.lora_r is not None:
        overrides.setdefault("model", {}).setdefault("lora", {})["r"] = args.lora_r
    if args.lora_alpha is not None:
        overrides.setdefault("model", {}).setdefault("lora", {})["alpha"] = args.lora_alpha
    if args.lora_target_modules:
        raw = args.lora_target_modules.strip()
        if raw.lower() == "all-linear":
            modules = list(ALL_LINEAR_MODULES)
        else:
            modules = [m.strip() for m in raw.split(",") if m.strip()]
        overrides.setdefault("model", {}).setdefault("lora", {})["target_modules"] = modules
    if args.learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.eval_max_new_tokens is not None:
        overrides.setdefault("evaluation", {})["max_new_tokens"] = args.eval_max_new_tokens
    if args.eval_batch_size is not None:
        overrides.setdefault("evaluation", {})["batch_size"] = args.eval_batch_size
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="由 experiment_config.yaml 驅動的 GPG vs Cooldown 對照實驗")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="集中設定檔路徑")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="覆蓋基礎模型路徑或 HF repo id")
    parser.add_argument("--hf_repo_id", type=str, default=None, help="覆蓋 model.hf_repo_id（搭配 --auto_download）")
    parser.add_argument("--run_tag", type=str, default=None, help="覆蓋 experiments.run_tag（避免覆蓋舊實驗）")
    parser.add_argument("--max_steps", type=int, default=None, help="覆蓋每組訓練總步數")
    parser.add_argument("--num_generations", type=int, default=None, help="覆蓋 G (training.num_generations)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None, help="覆蓋 training.per_device_train_batch_size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="覆蓋 training.gradient_accumulation_steps")
    parser.add_argument("--max_completion_length", type=int, default=None, help="覆蓋 L (training.max_completion_length)")
    parser.add_argument("--seed", type=int, default=None, help="覆蓋 training.seed")
    parser.add_argument("--grad_ckpt", type=_parse_bool_flag, default=None, help="覆蓋 gradient_checkpointing (true/false)")
    parser.add_argument("--sample_strategy", type=str, default=None, help="覆蓋 gpg.sample_strategy (uniform/soft/hard/medium)")
    parser.add_argument("--cooldown_steps", type=int, default=None, help="覆蓋全對後冷卻步數")
    parser.add_argument("--cooldown_backoff", type=_parse_bool_flag, default=None, help="是否啟用冷卻指數退避 (true/false)")
    parser.add_argument("--cooldown_backoff_factor", type=float, default=None, help="覆蓋指數退避倍數（預設 2.0）")
    parser.add_argument("--cooldown_max_steps", type=int, default=None, help="覆蓋指數退避上限步數（0=無上限）")
    parser.add_argument("--include_safesoft", action="store_true", help="額外啟用 Safe-Soft + Cooldown 組")
    parser.add_argument("--only_cooldown", action="store_true", help="跳過 baseline_gpg，只跑 Cooldown 實驗組")
    parser.add_argument("--skip_base_eval", action="store_true", help="評測時跳過 Base 模型（節省時間）")
    parser.add_argument("--precision", type=str, default=None, help="覆蓋 hardware.precision (auto/bf16/fp16/fp32)")
    parser.add_argument("--gpu_profile", type=str, default=None, help="覆蓋 hardware.gpu_profile")
    parser.add_argument("--strict_env", action="store_true", help="套件或 Python/PyTorch 版本不符時直接中止")
    parser.add_argument("--auto_download", action="store_true", help="本地模型不存在時自動由 hf_repo_id 下載")
    parser.add_argument("--dry_run", action="store_true", help="僅檢查環境與顯示實際執行計畫，不訓練不評測")
    parser.add_argument("--load_in_4bit", type=_parse_bool_flag, default=None, help="覆蓋 model.quantization.load_in_4bit（QLoRA，需 bitsandbytes）")
    parser.add_argument("--lora_r", type=int, default=None, help="覆蓋 LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="覆蓋 LoRA alpha")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="逗號分隔的 LoRA 目標層，或 all-linear")
    parser.add_argument("--learning_rate", type=float, default=None, help="覆蓋 training.learning_rate")
    parser.add_argument("--eval_max_new_tokens", type=int, default=None, help="覆蓋評測時的 max_new_tokens（避免長推導被截斷）")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="覆蓋評測 batch size")
    parser.add_argument("--eval_only", action="store_true", help="不訓練，只評測已存在的 adapter（例如換 max_new_tokens 重評）")
    parser.add_argument("--report_suffix", type=str, default="", help="報告檔名後綴，重評時避免覆蓋原報告（例如 _eval512）")
    args = parser.parse_args()

    cli_overrides = _build_cli_overrides(args)
    resolved = load_and_resolve_config(
        args.config,
        cli_overrides=cli_overrides,
        strict_env=True if args.strict_env else None,
        quiet=False,
    )
    print_status_report(resolved)

    groups = resolve_experiment_groups(
        resolved,
        include_safesoft=args.include_safesoft,
        only_cooldown=args.only_cooldown,
    )
    if not groups:
        raise RuntimeError("experiments.groups 沒有任何啟用的實驗組別。")

    exp_cfg = resolved["experiments"]
    run_tag = str(exp_cfg.get("run_tag", "200"))
    report_md_path = _resolve_root_path(str(exp_cfg.get("report_md", f"outputs/comparison_report_{run_tag}.md")).format(run_tag=run_tag))
    report_json_path = _resolve_root_path(str(exp_cfg.get("report_json", f"outputs/comparison_report_{run_tag}.json")).format(run_tag=run_tag))
    max_steps = int(resolved["training"]["max_steps"])
    if args.report_suffix:
        report_md_path = report_md_path.with_name(report_md_path.stem + args.report_suffix + report_md_path.suffix)
        report_json_path = report_json_path.with_name(report_json_path.stem + args.report_suffix + report_json_path.suffix)

    if args.dry_run:
        print("\n=== [DRY RUN] 預計執行計畫（不會修改現有 outputs 或開始訓練） ===")
        print(f"Model      : {resolved['model']['name_or_path']} (LoRA={resolved['model']['lora']['enabled']}, 4bit={resolved['model']['quantization']['load_in_4bit']})")
        print(f"Dataset    : {resolved['dataset']['train_file']} / {resolved['dataset']['eval_file']}")
        print(f"LoRA       : r={resolved['model']['lora']['r']}, alpha={resolved['model']['lora']['alpha']}, targets={resolved['model']['lora']['target_modules']}")
        print(f"LR / Eval  : lr={resolved['training']['learning_rate']}, eval_max_new_tokens={resolved['evaluation']['max_new_tokens']}, eval_bs={resolved['evaluation']['batch_size']}")
        print(f"Report MD  : {report_md_path}")
        print(f"Report JSON: {report_json_path}")
        for grp in groups:
            trl_preview = build_trl_config_dict(resolved, grp["overrides"])
            print(
                f"  - [{grp['slug']}] {grp['label']}\n"
                f"      output_dir={grp['output_dir']} | max_steps={trl_preview['max_steps']} "
                f"| G={trl_preview['num_generations']} | bs={trl_preview['per_device_train_batch_size']} "
                f"| L={trl_preview['max_completion_length']} | seed={trl_preview['seed']} | dtype={trl_preview['torch_dtype']} "
                f"| weighted={trl_preview['weighted_sample']} | strategy={trl_preview['sample_strategy']} "
                f"| cooldown={trl_preview['cooldown_steps']} | backoff={trl_preview['cooldown_backoff']} "
                f"(factor={trl_preview['cooldown_backoff_factor']}, max={trl_preview['cooldown_max_steps']})"
            )
        return

    train_file, eval_file = ensure_datasets(resolved)
    model_source = ensure_model_available(resolved, auto_download=args.auto_download)
    env_snapshot = build_env_snapshot(resolved)

    for grp in groups:
        if args.eval_only:
            if not (grp["output_dir"] / "adapter_model.safetensors").exists():
                raise FileNotFoundError(f"--eval_only 需要已訓練好的 adapter，但找不到: {grp['output_dir']}")
            print(f"[EVAL-ONLY] 略過訓練 {grp['label']}，直接評測 {grp['output_dir']}")
            continue
        trl_dict = build_trl_config_dict(resolved, grp["overrides"])
        trl_dict["model_name_or_path"] = str(model_source)
        run_training_if_needed(
            name=grp["label"],
            output_dir=grp["output_dir"],
            trl_dict=trl_dict,
            env_snapshot=env_snapshot,
            max_steps=int(trl_dict["max_steps"]),
        )

    train_data = load_jsonl(train_file)
    eval_data = load_jsonl(eval_file)
    train_logs = {grp["label"]: parse_train_logs(grp["output_dir"]) for grp in groups}

    model_display = Path(str(resolved["model"]["name_or_path"])).name
    eval_targets: list[tuple[str, Path | None]] = []
    if not args.skip_base_eval:
        eval_targets.append((f"Base ({model_display})", None))
    eval_targets.extend((grp["label"], grp["output_dir"]) for grp in groups)
    eval_results = {}
    for label, adapter in eval_targets:
        eval_results[label] = {
            "train_200": evaluate_model(f"{label} [Train-{len(train_data)}]", model_source, adapter, train_data, resolved),
            "eval_100": evaluate_model(f"{label} [Eval-{len(eval_data)}]", model_source, adapter, eval_data, resolved),
        }

    md = render_comparison_report_md(
        resolved=resolved,
        train_count=len(train_data),
        eval_count=len(eval_data),
        train_logs=train_logs,
        eval_results=eval_results,
    )
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(md, encoding="utf-8")
    report_json_path.write_text(
        json.dumps(
            {
                "env_snapshot": env_snapshot,
                "experiment_config": resolved["effective_config"],
                "train_logs": train_logs,
                "eval": eval_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"報告已儲存至:\n- {report_md_path}\n- {report_json_path}")


if __name__ == "__main__":
    main()
