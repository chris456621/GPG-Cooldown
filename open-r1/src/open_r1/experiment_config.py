"""集中管理 GPG + Cooldown 實驗設定、Windows 11 套件檢查與跨 CUDA GPU 自動相容模組。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from packaging.version import Version

OPEN_R1_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = OPEN_R1_ROOT / "experiment_config.yaml"

_VRAM_PROFILE_THRESHOLDS = (
    (7.5, "vram_6gb"),
    (11.0, "vram_8gb"),
    (15.0, "vram_12gb"),
    (22.0, "vram_16gb"),
)


def is_master_config(data: dict[str, Any] | None) -> bool:
    """判斷 YAML 內容是否為集中設定檔（而非傳統平坦 TRL recipe）。"""
    if not isinstance(data, dict):
        return False
    return "schema_version" in data or ("model" in data and "training" in data)


def _has_package(pkg_name: str) -> bool:
    return importlib.util.find_spec(pkg_name) is not None


def _resolve_path_under_root(raw_path: str, root: Path = OPEN_R1_ROOT) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else (root / p)


def detect_hardware(device_index: int = 0) -> dict[str, Any]:
    """偵測目前作業系統、Python、PyTorch/CUDA 版本與 GPU 硬體資訊。"""
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    total_vram_gb = 0.0
    free_vram_gb = 0.0
    cc: tuple[int, int] | None = None
    bf16_native = False
    device_count = torch.cuda.device_count() if cuda_available else 0

    if cuda_available and device_count > 0:
        idx = min(device_index, device_count - 1)
        props = torch.cuda.get_device_properties(idx)
        gpu_name = props.name
        total_vram_gb = round(props.total_memory / (1024**3), 2)
        try:
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            free_vram_gb = round(free_bytes / (1024**3), 2)
        except Exception:
            free_vram_gb = total_vram_gb
        cc = (int(props.major), int(props.minor))
        try:
            bf16_native = bool(torch.cuda.is_bf16_supported(including_emulation=False))
        except TypeError:
            bf16_native = cc[0] >= 8

    os_name = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    if os_name == "Windows" and os_release == "10":
        try:
            build_num = int(os_version.split(".")[2])
            if build_num >= 22000:
                os_release = "11"
        except (IndexError, ValueError):
            pass

    return {
        "os": os_name,
        "os_release": os_release,
        "os_version": os_version,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "free_vram_gb": free_vram_gb,
        "compute_capability": list(cc) if cc is not None else None,
        "bf16_native": bf16_native,
        "has_bitsandbytes": _has_package("bitsandbytes"),
        "has_flash_attn": _has_package("flash_attn"),
    }


def select_gpu_profile(hw: dict[str, Any], requested_profile: str) -> str:
    """依 GPU 總 VRAM 選擇對應硬體檔位。"""
    if requested_profile and requested_profile not in ("auto", "manual"):
        return requested_profile
    if requested_profile == "manual":
        return "manual"
    if not hw.get("cuda_available"):
        return "vram_6gb"

    vram_gb = float(hw.get("total_vram_gb") or 0.0)
    for limit_gb, profile_name in _VRAM_PROFILE_THRESHOLDS:
        if vram_gb < limit_gb:
            return profile_name
    return "vram_24gb_plus"


def resolve_precision(hw: dict[str, Any], requested: str) -> tuple[str, bool, bool, list[str]]:
    """依 GPU Compute Capability 解析數值精度，回傳 (torch_dtype, bf16, fp16, warnings)。"""
    warnings: list[str] = []
    req = (requested or "auto").lower()
    cuda_ok = bool(hw.get("cuda_available"))
    cc = hw.get("compute_capability")
    cc_major = cc[0] if cc else 0
    bf16_native = bool(hw.get("bf16_native")) or cc_major >= 8

    if not cuda_ok:
        if req in ("bf16", "fp16"):
            warnings.append(f"未偵測到 CUDA GPU，已將精度從 {req} 自動降級為 fp32。")
        return "float32", False, False, warnings

    if req == "auto":
        if bf16_native:
            return "bfloat16", True, False, warnings
        return "float16", False, True, warnings

    if req == "bf16":
        if not bf16_native:
            warnings.append(
                f"GPU ({hw.get('gpu_name')}, CC={cc}) 不支援原生 bf16，已自動降級為 fp16 避免數值模擬拖慢或報錯。"
            )
            return "float16", False, True, warnings
        return "bfloat16", True, False, warnings

    if req == "fp16":
        return "float16", False, True, warnings

    if req == "fp32":
        return "float32", False, False, warnings

    raise ValueError(f"不支援的 hardware.precision 設定: {requested!r}（僅支援 auto/bf16/fp16/fp32）")


def resolve_attn_implementation(hw: dict[str, Any], requested: str) -> tuple[str, list[str]]:
    """依 OS 與安裝套件解析 Attention 實作方式。"""
    warnings: list[str] = []
    req = (requested or "auto").lower()
    is_windows = str(hw.get("os", "")).lower().startswith("win")
    has_flash = bool(hw.get("has_flash_attn"))
    cc = hw.get("compute_capability")
    cc_major = cc[0] if cc else 0

    if req == "auto":
        if not is_windows and has_flash and cc_major >= 8:
            return "flash_attention_2", warnings
        return "sdpa", warnings

    if req == "flash_attention_2":
        if is_windows or not has_flash or cc_major < 8:
            warnings.append(
                "要求使用 flash_attention_2，但目前環境不支援（Windows 或未安裝 flash_attn / CC < 8.0），已退回 sdpa。"
            )
            return "sdpa", warnings
        return "flash_attention_2", warnings

    if req in ("sdpa", "eager"):
        return req, warnings

    raise ValueError(f"不支援的 hardware.attn_implementation 設定: {requested!r}")


def _base_version(ver_str: str) -> str:
    return ver_str.split("+", 1)[0].strip()


def format_win11_setup_commands(cfg: dict[str, Any]) -> str:
    """從設定檔產生 Windows 11 PowerShell 一鍵安裝 / 修復指令。"""
    env_cfg = cfg.get("environment", {})
    torch_ver = env_cfg.get("torch_version", "2.5.1")
    torch_url = env_cfg.get("torch_cuda_index_url", "https://download.pytorch.org/whl/cu124")
    pkgs = env_cfg.get("packages", {})
    pkg_specs = " `\n  ".join(f"{name}=={ver}" for name, ver in pkgs.items())

    lines = [
        "# Windows 11 PowerShell 環境安裝 / 同步指令（請勿執行 pip install -e .）",
        ".\\openr1_venv\\Scripts\\python.exe -m pip install --upgrade pip",
        f".\\openr1_venv\\Scripts\\pip.exe install torch=={torch_ver} --index-url {torch_url}",
    ]
    if pkg_specs:
        lines.append(f".\\openr1_venv\\Scripts\\pip.exe install `\n  {pkg_specs}")
    return "\n".join(lines)


def check_environment(cfg: dict[str, Any], hw: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    """核對 Python、PyTorch 與所有鎖定套件版本是否符合 environment 規範。"""
    env_cfg = cfg.get("environment", {})
    items: list[dict[str, str]] = []
    mismatches: list[str] = []

    py_expected = str(env_cfg.get("python_minor", "")).strip()
    py_actual = ".".join(str(hw.get("python_version", "")).split(".")[:2])
    if py_expected:
        ok = py_actual == py_expected
        items.append(
            {
                "name": "python",
                "expected": f"{py_expected}.x",
                "actual": str(hw.get("python_version", "")),
                "status": "OK" if ok else "MISMATCH",
            }
        )
        if not ok:
            mismatches.append(f"python 預期 {py_expected}.x，目前為 {hw.get('python_version')}")

    torch_expected = str(env_cfg.get("torch_version", "")).strip()
    torch_actual_raw = str(hw.get("torch_version", ""))
    torch_actual = _base_version(torch_actual_raw)
    if torch_expected:
        ok = torch_actual == torch_expected
        items.append(
            {
                "name": "torch",
                "expected": torch_expected,
                "actual": torch_actual_raw,
                "status": "OK" if ok else "MISMATCH",
            }
        )
        if not ok:
            mismatches.append(f"torch 預期 {torch_expected}，目前為 {torch_actual_raw}")

    if not hw.get("cuda_available"):
        mismatches.append("torch.cuda.is_available() 為 False，請確認是否安裝 CUDA 版 PyTorch 與 NVIDIA 驅動。")

    for pkg_name, expected_ver in env_cfg.get("packages", {}).items():
        expected_str = str(expected_ver).strip()
        try:
            actual_ver = importlib.metadata.version(pkg_name)
            ok = Version(_base_version(actual_ver)) == Version(_base_version(expected_str))
            status = "OK" if ok else "MISMATCH"
            if not ok:
                mismatches.append(f"{pkg_name} 預期 {expected_str}，目前為 {actual_ver}")
        except importlib.metadata.PackageNotFoundError:
            actual_ver = "NOT_INSTALLED"
            status = "MISSING"
            mismatches.append(f"{pkg_name} 尚未安裝（預期 {expected_str}）")
        items.append(
            {
                "name": pkg_name,
                "expected": expected_str,
                "actual": actual_ver,
                "status": status,
            }
        )

    report = {
        "passed": len(mismatches) == 0,
        "items": items,
        "mismatches": mismatches,
        "fix_commands": format_win11_setup_commands(cfg),
    }

    if strict and mismatches:
        msg = "環境版本檢查未通過（strict=True）：\n- " + "\n- ".join(mismatches) + "\n\n建議修復指令：\n" + report["fix_commands"]
        raise RuntimeError(msg)

    return report


def apply_env_vars(cfg: dict[str, Any], root: Path = OPEN_R1_ROOT) -> dict[str, str]:
    """套用設定檔內的環境變數與 CUDA_VISIBLE_DEVICES。"""
    applied: dict[str, str] = {}
    env_vars = cfg.get("environment", {}).get("env_vars", {}) or {}
    for key, val in env_vars.items():
        if val is None:
            continue
        str_val = str(val)
        if key == "PYTHONPATH":
            resolved_src = str(_resolve_path_under_root(str_val, root))
            existing = os.environ.get("PYTHONPATH", "")
            parts = [p for p in existing.split(os.pathsep) if p] if existing else []
            if resolved_src not in parts:
                str_val = os.pathsep.join([resolved_src, *parts]) if parts else resolved_src
            else:
                str_val = existing
        os.environ[key] = str_val
        applied[key] = str_val

    cuda_devs = cfg.get("hardware", {}).get("cuda_visible_devices")
    if cuda_devs is not None and str(cuda_devs).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devs)
        applied["CUDA_VISIBLE_DEVICES"] = str(cuda_devs)

    return applied


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _validate_batch_and_generations(training: dict[str, Any]) -> None:
    batch_size = int(training["per_device_train_batch_size"])
    num_gen = int(training["num_generations"])
    if batch_size < num_gen or batch_size % num_gen != 0:
        raise ValueError(
            f"per_device_train_batch_size ({batch_size}) 必須大於等於且可整除 num_generations ({num_gen})。"
        )


def _format_template(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        formatted = value.format(**variables)
        if value.startswith("{") and value.endswith("}") and formatted.lstrip("-").isdigit():
            return int(formatted)
        return formatted
    if isinstance(value, dict):
        return {k: _format_template(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_format_template(v, variables) for v in value]
    return value


def load_and_resolve_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    cli_overrides: dict[str, Any] | None = None,
    strict_env: bool | None = None,
    quiet: bool = False,
    hw_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """讀取集中設定檔、檢查環境並依硬體解析最終執行參數。"""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = OPEN_R1_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到設定檔: {cfg_path}")

    raw_bytes = cfg_path.read_bytes()
    config_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw_cfg = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not is_master_config(raw_cfg):
        raise ValueError(f"{cfg_path} 不是集中式 experiment_config 格式。")

    cfg = copy.deepcopy(raw_cfg)
    if cli_overrides:
        cfg = _deep_merge(cfg, cli_overrides)

    apply_env_vars(cfg, root=OPEN_R1_ROOT)
    hw = copy.deepcopy(hw_override) if hw_override is not None else detect_hardware()

    use_strict = bool(cfg.get("environment", {}).get("strict_env_check", False)) if strict_env is None else bool(strict_env)
    env_check = check_environment(cfg, hw, strict=use_strict)

    hw_cfg = cfg.get("hardware", {})
    warnings: list[str] = []

    profile_name = select_gpu_profile(hw, str(hw_cfg.get("gpu_profile", "auto")))
    profiles_map = hw_cfg.get("gpu_profiles", {}) or {}
    profile_data = profiles_map.get(profile_name, {}) if profile_name != "manual" else {}

    torch_dtype, bf16_flag, fp16_flag, prec_warns = resolve_precision(hw, str(hw_cfg.get("precision", "auto")))
    warnings.extend(prec_warns)

    attn_impl, attn_warns = resolve_attn_implementation(hw, str(hw_cfg.get("attn_implementation", "auto")))
    warnings.extend(attn_warns)

    use_vllm = bool(hw_cfg.get("use_vllm", False))
    if str(hw.get("os", "")).lower().startswith("win") and use_vllm:
        warnings.append("Windows 不支援 vLLM，已強制將 use_vllm 設為 False。")
        use_vllm = False

    training = copy.deepcopy(cfg["training"])
    if "gradient_checkpointing" in profile_data and hw_cfg.get("gpu_profile", "auto") != "manual":
        training["gradient_checkpointing"] = bool(profile_data["gradient_checkpointing"])

    auto_adjust = bool(hw_cfg.get("auto_adjust_training_hparams", False))
    if auto_adjust and profile_data.get("training_overrides"):
        training = _deep_merge(training, profile_data["training_overrides"])

    _validate_batch_and_generations(training)

    quant_cfg = cfg.get("model", {}).get("quantization", {}) or {}
    if quant_cfg.get("load_in_4bit") and not hw.get("has_bitsandbytes"):
        raise RuntimeError("已啟用 model.quantization.load_in_4bit=true，但環境未安裝 bitsandbytes。")

    eval_cfg = copy.deepcopy(cfg.get("evaluation", {}))
    raw_eval_bs = eval_cfg.get("batch_size", "auto")
    if str(raw_eval_bs).lower() == "auto":
        eval_batch_size = int(profile_data.get("eval_batch_size", 16))
    else:
        eval_batch_size = int(raw_eval_bs)

    resolved_hardware = {
        "profile": profile_name,
        "torch_dtype": torch_dtype,
        "bf16": bf16_flag,
        "fp16": fp16_flag,
        "attn_implementation": attn_impl,
        "use_vllm": use_vllm,
        "eval_batch_size": eval_batch_size,
        "auto_adjusted_training_hparams": auto_adjust,
    }

    resolved_evaluation = {
        "do_sample": bool(eval_cfg.get("do_sample", False)),
        "max_new_tokens": int(eval_cfg.get("max_new_tokens", 256)),
        "batch_size": eval_batch_size,
    }
    effective_config = {
        "schema_version": cfg.get("schema_version", 1),
        "environment": copy.deepcopy(cfg.get("environment", {})),
        "hardware": copy.deepcopy(hw_cfg),
        "model": copy.deepcopy(cfg["model"]),
        "dataset": copy.deepcopy(cfg["dataset"]),
        "training": copy.deepcopy(training),
        "gpg": copy.deepcopy(cfg["gpg"]),
        "evaluation": copy.deepcopy(resolved_evaluation),
        "experiments": copy.deepcopy(cfg.get("experiments", {})),
    }

    resolved = {
        "config_path": str(cfg_path),
        "config_sha256": config_sha256,
        "root_dir": str(OPEN_R1_ROOT),
        "hardware_detected": hw,
        "hardware_resolved": resolved_hardware,
        "env_check": env_check,
        "warnings": warnings,
        "environment": cfg.get("environment", {}),
        "hardware": hw_cfg,
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "training": training,
        "gpg": cfg["gpg"],
        "evaluation": resolved_evaluation,
        "experiments": cfg.get("experiments", {}),
        "effective_config": effective_config,
    }

    if not quiet:
        for w in warnings:
            print(f"[WARN] {w}")
        if not env_check["passed"]:
            print("[WARN] 環境檢查發現不一致項目：")
            for item in env_check["mismatches"]:
                print(f"  - {item}")

    return resolved


def build_trl_config_dict(resolved: dict[str, Any], group_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """將解析後設定轉為僅含 TrlParser((GRPOScriptArguments, GPGConfig, ModelConfig)) 合法欄位的平坦 dict。"""
    model_cfg = resolved["model"]
    lora_cfg = model_cfg.get("lora", {}) or {}
    quant_cfg = model_cfg.get("quantization", {}) or {}
    ds_cfg = resolved["dataset"]
    tr_cfg = copy.deepcopy(resolved["training"])
    gpg_cfg = copy.deepcopy(resolved["gpg"])
    hw_res = resolved["hardware_resolved"]

    model_path = str(_resolve_path_under_root(str(model_cfg["name_or_path"])))
    dataset_name = str(_resolve_path_under_root(str(ds_cfg["name"])))

    trl_dict: dict[str, Any] = {
        # ModelConfig
        "model_name_or_path": model_path,
        "model_revision": str(model_cfg.get("revision", "main")),
        "torch_dtype": str(hw_res["torch_dtype"]),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "attn_implementation": str(hw_res["attn_implementation"]),
        "use_peft": bool(lora_cfg.get("enabled", True)),
        "lora_r": int(lora_cfg.get("r", 8)),
        "lora_alpha": int(lora_cfg.get("alpha", 16)),
        "lora_dropout": float(lora_cfg.get("dropout", 0.05)),
        "lora_target_modules": list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
        "load_in_4bit": bool(quant_cfg.get("load_in_4bit", False)),
        "bnb_4bit_quant_type": str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
        "use_bnb_nested_quant": bool(quant_cfg.get("use_bnb_nested_quant", False)),
        # GRPOScriptArguments
        "dataset_name": dataset_name,
        "reward_funcs": list(gpg_cfg.get("reward_funcs", ["accuracy"])),
        "reward_weights": [float(x) for x in gpg_cfg.get("reward_weights", [1.0])],
        # GRPOConfig / TrainingArguments
        "system_prompt": ds_cfg.get("system_prompt"),
        "bf16": bool(hw_res["bf16"]),
        "fp16": bool(hw_res["fp16"]),
        "use_vllm": bool(hw_res["use_vllm"]),
        "do_eval": bool(tr_cfg.get("do_eval", False)),
        "eval_strategy": str(tr_cfg.get("eval_strategy", "no")),
        "gradient_accumulation_steps": int(tr_cfg["gradient_accumulation_steps"]),
        "gradient_checkpointing": bool(tr_cfg["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {
            "use_reentrant": bool(tr_cfg.get("gradient_checkpointing_use_reentrant", False))
        },
        "learning_rate": float(tr_cfg["learning_rate"]),
        "log_completions": bool(tr_cfg.get("log_completions", False)),
        "log_level": str(tr_cfg.get("log_level", "info")),
        "logging_first_step": bool(tr_cfg.get("logging_first_step", True)),
        "logging_steps": int(tr_cfg.get("logging_steps", 1)),
        "logging_strategy": str(tr_cfg.get("logging_strategy", "steps")),
        "lr_scheduler_type": str(tr_cfg["lr_scheduler_type"]),
        "max_prompt_length": int(tr_cfg["max_prompt_length"]),
        "max_completion_length": int(tr_cfg["max_completion_length"]),
        "max_steps": int(tr_cfg["max_steps"]),
        "num_generations": int(tr_cfg["num_generations"]),
        "num_train_epochs": int(tr_cfg["num_train_epochs"]),
        "output_dir": str(_resolve_path_under_root("outputs/default_run")),
        "overwrite_output_dir": bool(tr_cfg.get("overwrite_output_dir", True)),
        "per_device_train_batch_size": int(tr_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(tr_cfg.get("per_device_eval_batch_size", 8)),
        "push_to_hub": bool(tr_cfg.get("push_to_hub", False)),
        "report_to": list(tr_cfg.get("report_to", ["tensorboard"])),
        "save_strategy": str(tr_cfg.get("save_strategy", "steps")),
        "save_steps": int(tr_cfg.get("save_steps", 50)),
        "save_total_limit": int(tr_cfg.get("save_total_limit", 2)),
        "seed": int(tr_cfg["seed"]),
        "warmup_ratio": float(tr_cfg["warmup_ratio"]),
        # GPGConfig
        "scale_rewards": bool(gpg_cfg.get("scale_rewards", False)),
        "min_inverse_alpha": float(gpg_cfg.get("min_inverse_alpha", 0.0)),
        "weighted_sample": bool(gpg_cfg.get("weighted_sample", True)),
        "sample_strategy": str(gpg_cfg.get("sample_strategy", "uniform")),
        "cooldown_steps": int(gpg_cfg.get("cooldown_steps", 20)),
        "scale_batch": bool(gpg_cfg.get("scale_batch", False)),
        "adjust_lr": bool(gpg_cfg.get("adjust_lr", False)),
        "adjust_gd": bool(gpg_cfg.get("adjust_gd", False)),
    }

    if group_overrides:
        allowed_keys = set(trl_dict.keys())
        unknown = set(group_overrides.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"group overrides 包含未知欄位: {sorted(unknown)}")
        for k, v in group_overrides.items():
            if k == "output_dir" and v is not None:
                trl_dict[k] = str(_resolve_path_under_root(str(v)))
            else:
                trl_dict[k] = v

    _validate_batch_and_generations(trl_dict)
    return trl_dict


def resolve_experiment_groups(resolved: dict[str, Any], include_safesoft: bool = False) -> list[dict[str, Any]]:
    """依 experiments 設定展開每組實驗名稱、輸出目錄與 overrides。"""
    exp_cfg = resolved.get("experiments", {})
    run_tag = str(exp_cfg.get("run_tag", "200"))
    cooldown_steps = int(resolved["gpg"].get("cooldown_steps", 20))
    out_template = str(exp_cfg.get("output_dir_template", "outputs/exp{run_tag}_{slug}"))

    expanded: list[dict[str, Any]] = []
    for grp in exp_cfg.get("groups", []):
        slug = str(grp["slug"])
        enabled = bool(grp.get("enabled", True))
        if include_safesoft and slug == "safesoft_cooldown":
            enabled = True
        if not enabled:
            continue

        fmt_vars = {
            "run_tag": run_tag,
            "slug": slug,
            "cooldown_steps": cooldown_steps,
        }
        label = _format_template(str(grp.get("label", slug)), fmt_vars)
        out_dir = _resolve_path_under_root(_format_template(out_template, fmt_vars))
        overrides = _format_template(grp.get("overrides", {}) or {}, fmt_vars)
        overrides["output_dir"] = str(out_dir)

        expanded.append(
            {
                "slug": slug,
                "label": label,
                "output_dir": out_dir,
                "overrides": overrides,
            }
        )
    return expanded


def export_trl_yaml(trl_dict: dict[str, Any], out_path: Path | str) -> Path:
    """將平坦化 TRL 設定寫出為 YAML 檔（UTF-8）。"""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(trl_dict, f, allow_unicode=True, sort_keys=False)
    return target


def build_env_snapshot(resolved: dict[str, Any]) -> dict[str, Any]:
    """組裝完整環境與硬體指紋，供報告與 checkpoint 保存存證。"""
    return {
        "config_path": resolved["config_path"],
        "config_sha256": resolved["config_sha256"],
        "hardware_detected": resolved["hardware_detected"],
        "hardware_resolved": resolved["hardware_resolved"],
        "env_check": {
            "passed": resolved["env_check"]["passed"],
            "mismatches": resolved["env_check"]["mismatches"],
            "items": resolved["env_check"]["items"],
        },
        "warnings": resolved["warnings"],
        "model": resolved["model"],
        "training": resolved["training"],
        "gpg": resolved["gpg"],
        "evaluation": resolved["evaluation"],
        "effective_config": resolved["effective_config"],
    }


def prepare_gpg_cli_args(argv: list[str]) -> list[str]:
    """若 gpg.py 的 --config 指向集中設定檔，解析並轉為平坦 TRL YAML 後回傳新 argv。"""
    args = list(argv)
    cfg_idx: int | None = None
    cfg_val: str | None = None

    for i, tok in enumerate(args):
        if tok == "--config" and i + 1 < len(args):
            cfg_idx = i + 1
            cfg_val = args[i + 1]
            break
        if tok.startswith("--config="):
            cfg_idx = i
            cfg_val = tok.split("=", 1)[1]
            break

    if cfg_val is None:
        return args

    cfg_path = _resolve_path_under_root(cfg_val)
    if not cfg_path.exists():
        return args

    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not is_master_config(loaded):
        return args

    resolved = load_and_resolve_config(cfg_path)
    trl_dict = build_trl_config_dict(resolved)
    tmp_yaml = OPEN_R1_ROOT / "outputs" / "_last_resolved_trl_config.yaml"
    export_trl_yaml(trl_dict, tmp_yaml)

    if args[cfg_idx].startswith("--config="):
        args[cfg_idx] = f"--config={tmp_yaml}"
    else:
        args[cfg_idx] = str(tmp_yaml)
    return args


def sync_win11_packages(cfg: dict[str, Any]) -> None:
    """依設定檔鎖定版本在目前 Python 環境中執行 pip 安裝（不呼叫 pip install -e .）。"""
    env_cfg = cfg.get("environment", {})
    torch_ver = str(env_cfg.get("torch_version", "2.5.1"))
    torch_url = str(env_cfg.get("torch_cuda_index_url", "https://download.pytorch.org/whl/cu124"))
    pkgs = [f"{name}=={ver}" for name, ver in (env_cfg.get("packages", {}) or {}).items()]

    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", f"torch=={torch_ver}", "--index-url", torch_url],
        check=True,
    )
    if pkgs:
        subprocess.run([sys.executable, "-m", "pip", "install", *pkgs], check=True)


def print_status_report(resolved: dict[str, Any]) -> None:
    hw = resolved["hardware_detected"]
    hr = resolved["hardware_resolved"]
    ec = resolved["env_check"]
    print("=== 硬體與環境檢查摘要 ===")
    print(f"OS / Python : {hw['os']} {hw['os_release']} | Python {hw['python_version']}")
    print(f"PyTorch/CUDA: {hw['torch_version']} | CUDA={hw['cuda_version']} | available={hw['cuda_available']}")
    print(
        f"GPU         : {hw['gpu_name']} | VRAM={hw['total_vram_gb']} GB (free {hw['free_vram_gb']} GB) "
        f"| CC={hw['compute_capability']} | bf16_native={hw['bf16_native']}"
    )
    print(
        f"Resolved    : profile={hr['profile']}, dtype={hr['torch_dtype']} (bf16={hr['bf16']}, fp16={hr['fp16']}), "
        f"attn={hr['attn_implementation']}, eval_bs={hr['eval_batch_size']}, "
        f"auto_adjust_hparams={hr['auto_adjusted_training_hparams']}"
    )
    print(f"Config SHA  : {resolved['config_sha256'][:16]}...")
    print("--- 套件版本核對 ---")
    for item in ec["items"]:
        print(f"  [{item['status']:8s}] {item['name']:24s} expected={item['expected']:10s} actual={item['actual']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPG + Cooldown 集中設定與環境管理工具")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="集中設定檔路徑")
    parser.add_argument("--check", action="store_true", help="執行硬體偵測與套件版本核對")
    parser.add_argument("--strict", action="store_true", help="套件版本不符時直接以錯誤結束")
    parser.add_argument("--print-setup", action="store_true", help="印出 Windows 11 環境建置 / 修復指令")
    parser.add_argument("--sync-env", action="store_true", help="依設定檔自動安裝鎖定的 PyTorch 與套件")
    parser.add_argument("--export-trl", type=str, default=None, help="將解析後設定匯出為平坦 TRL YAML")
    args = parser.parse_args()

    cfg_path = _resolve_path_under_root(args.config)
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    if args.sync_env:
        sync_win11_packages(raw_cfg)

    if args.print_setup:
        print(format_win11_setup_commands(raw_cfg))

    if args.check or args.export_trl or (not args.print_setup and not args.sync_env):
        resolved = load_and_resolve_config(
            cfg_path,
            strict_env=True if args.strict else None,
            quiet=False,
        )
        print_status_report(resolved)
        if args.export_trl:
            out = export_trl_yaml(build_trl_config_dict(resolved), Path(args.export_trl))
            print(f"[EXPORT] 已輸出 TRL 設定至: {out}")


if __name__ == "__main__":
    main()
