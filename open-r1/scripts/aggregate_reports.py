"""
團隊實驗結果總彙整腳本：
掃描所有組員回傳的 outputs/comparison_report_*.json，檢查控制變因是否一致，
並一次輸出 summary_all_experiments.md 與 summary_all_experiments.csv（UTF-8 BOM，Windows Excel 可直接開）。

用法（於 C:\\Users\\chris\\Desktop\\GPG\\open-r1 執行）：
    .\\openr1_venv\\Scripts\\python.exe scripts/aggregate_reports.py
    .\\openr1_venv\\Scripts\\python.exe scripts/aggregate_reports.py --input_dir outputs --output_dir outputs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _infer_run_tag(path: Path) -> str:
    stem = path.stem
    prefix = "comparison_report_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _find_base_entry(eval_map: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    for label, res in eval_map.items():
        if label.startswith("Base (") or label.lower().startswith("base"):
            return label, res
    return None, None


def _get_eval_split(entry: dict[str, Any] | None, preferred_prefix: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    if preferred_prefix in entry:
        return entry[preferred_prefix]
    for k, v in entry.items():
        if k.startswith(preferred_prefix.split("_", 1)[0]):
            return v
    return {}


def _fmt_acc(split_res: dict[str, Any]) -> str:
    if not split_res or "accuracy" not in split_res:
        return "N/A"
    acc = float(split_res["accuracy"]) * 100.0
    return f"{acc:.1f}% ({split_res.get('correct', '?')}/{split_res.get('total', '?')})"


def _match_train_log(label: str, train_logs: dict[str, Any]) -> dict[str, Any]:
    if label in train_logs:
        return train_logs[label]
    return {}


def _extract_row_meta(report_path: Path, data: dict[str, Any]) -> dict[str, Any]:
    run_tag = _infer_run_tag(report_path)
    env_snap = data.get("env_snapshot") or {}
    exp_cfg = data.get("experiment_config") or env_snap.get("effective_config") or {}
    hw_det = env_snap.get("hardware_detected") or {}
    hw_res = env_snap.get("hardware_resolved") or {}

    model_cfg = exp_cfg.get("model") or env_snap.get("model") or {}
    tr_cfg = exp_cfg.get("training") or env_snap.get("training") or {}
    gpg_cfg = exp_cfg.get("gpg") or env_snap.get("gpg") or {}
    ds_cfg = exp_cfg.get("dataset") or {}
    ev_cfg = exp_cfg.get("evaluation") or env_snap.get("evaluation") or {}
    lora_cfg = model_cfg.get("lora") or {}

    raw_model = str(model_cfg.get("name_or_path") or "")
    model_name = Path(raw_model).name if raw_model else "N/A"
    if model_name == "N/A":
        base_lbl, _ = _find_base_entry(data.get("eval") or {})
        if base_lbl and "(" in base_lbl and ")" in base_lbl:
            model_name = base_lbl.split("(", 1)[1].rsplit(")", 1)[0]

    gpu_name = hw_det.get("gpu_name") or "N/A"
    backend = hw_det.get("accelerator_backend") or ("cuda" if hw_det.get("cuda_available") else "N/A")
    os_name = f"{hw_det.get('os', '')} {hw_det.get('os_release', '')}".strip() or "N/A"

    return {
        "run_tag": run_tag,
        "report_file": report_path.name,
        "model": model_name,
        "gpu": gpu_name,
        "backend": backend,
        "os": os_name,
        "dtype": hw_res.get("torch_dtype", "N/A"),
        "G": tr_cfg.get("num_generations", "N/A"),
        "L": tr_cfg.get("max_completion_length", "N/A"),
        "steps": tr_cfg.get("max_steps", "N/A"),
        "batch_size": tr_cfg.get("per_device_train_batch_size", "N/A"),
        "grad_accum": tr_cfg.get("gradient_accumulation_steps", "N/A"),
        "seed": tr_cfg.get("seed", "N/A"),
        "lr": tr_cfg.get("learning_rate", "N/A"),
        "lora_r": lora_cfg.get("r", "N/A"),
        "dataset_seed": ds_cfg.get("seed", "N/A"),
        "max_prompt_len": tr_cfg.get("max_prompt_length", "N/A"),
        "eval_max_tokens": ev_cfg.get("max_new_tokens", "N/A"),
        "cd": gpg_cfg.get("cooldown_steps", "N/A"),
        "backoff": gpg_cfg.get("cooldown_backoff", False if gpg_cfg else "N/A"),
        "backoff_factor": gpg_cfg.get("cooldown_backoff_factor", "N/A"),
        "strategy": gpg_cfg.get("sample_strategy", "N/A"),
        "config_sha": str(env_snap.get("config_sha256", ""))[:12] or "N/A",
    }


def collect_reports(input_dir: Path, pattern: str) -> tuple[list[dict[str, Any]], list[str]]:
    files = sorted(input_dir.glob(pattern))
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    control_tracker: dict[str, set[str]] = {
        "lr": set(),
        "lora_r": set(),
        "max_prompt_len": set(),
        "eval_max_tokens": set(),
    }
    seen_backends: set[str] = set()

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"讀取失敗 {path.name}: {exc}")
            continue

        meta = _extract_row_meta(path, data)
        for key in control_tracker:
            val = meta.get(key)
            if val != "N/A":
                control_tracker[key].add(str(val))
        if meta["backend"] != "N/A":
            seen_backends.add(str(meta["backend"]))

        eval_map = data.get("eval") or {}
        train_logs = data.get("train_logs") or {}
        _, base_res = _find_base_entry(eval_map)
        base_eval_split = _get_eval_split(base_res, "eval_100")
        base_eval_acc = float(base_eval_split["accuracy"]) if "accuracy" in base_eval_split else None

        if not eval_map and train_logs:
            for lbl, log in train_logs.items():
                rows.append(
                    {
                        **meta,
                        "variant": lbl,
                        "runtime_sec": log.get("runtime_sec", "N/A"),
                        "avg_reward": log.get("avg_reward", "N/A"),
                        "last_q_reward": log.get("last_quarter_reward", "N/A"),
                        "easy": log.get("avg_easy_per_step", "N/A"),
                        "inv_alpha": log.get("avg_valid_grad_ratio", "N/A"),
                        "nonzero_grad": log.get("nonzero_grad_steps", "N/A"),
                        "train_acc_str": "N/A",
                        "eval_acc_str": "N/A",
                        "train_acc": "",
                        "eval_acc": "",
                        "delta_eval_pp": "N/A",
                    }
                )
            continue

        for label, res in eval_map.items():
            log = _match_train_log(label, train_logs)
            tr_split = _get_eval_split(res, "train_200")
            ev_split = _get_eval_split(res, "eval_100")
            ev_acc = float(ev_split["accuracy"]) if "accuracy" in ev_split else None
            tr_acc = float(tr_split["accuracy"]) if "accuracy" in tr_split else None

            if label.startswith("Base (") or base_eval_acc is None or ev_acc is None:
                delta_str = "-" if label.startswith("Base (") else "N/A"
            else:
                delta_pp = (ev_acc - base_eval_acc) * 100.0
                delta_str = f"{delta_pp:+.1f} pp"

            rows.append(
                {
                    **meta,
                    "variant": label,
                    "runtime_sec": log.get("runtime_sec", "-"),
                    "avg_reward": f"{log['avg_reward']:.4f}" if isinstance(log.get("avg_reward"), (int, float)) else "-",
                    "last_q_reward": (
                        f"{log['last_quarter_reward']:.4f}"
                        if isinstance(log.get("last_quarter_reward"), (int, float))
                        else "-"
                    ),
                    "easy": f"{log['avg_easy_per_step']:.3f}" if isinstance(log.get("avg_easy_per_step"), (int, float)) else "-",
                    "inv_alpha": (
                        f"{log['avg_valid_grad_ratio']:.3f}"
                        if isinstance(log.get("avg_valid_grad_ratio"), (int, float))
                        else "-"
                    ),
                    "nonzero_grad": log.get("nonzero_grad_steps", "-"),
                    "train_acc_str": _fmt_acc(tr_split),
                    "eval_acc_str": _fmt_acc(ev_split),
                    "train_acc": f"{tr_acc:.4f}" if tr_acc is not None else "",
                    "eval_acc": f"{ev_acc:.4f}" if ev_acc is not None else "",
                    "delta_eval_pp": delta_str,
                }
            )

    for key, vals in control_tracker.items():
        if len(vals) > 1:
            warnings.append(f"控制變因 `{key}` 在不同報告間不一致：{sorted(vals)}")
    if "mps" in seen_backends and "cuda" in seen_backends:
        warnings.append("偵測到同時存在 MPS (Mac M1) 與 CUDA (NVIDIA) 報告：兩者運算精度/後端不同，耗時 (`runtime_sec`) 不可直接橫比。")

    return rows, warnings


def write_summary(rows: list[dict[str, Any]], warnings: list[str], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "summary_all_experiments.md"
    csv_path = output_dir / "summary_all_experiments.csv"

    md_lines = [
        "# GPG + Cooldown 團隊實驗結果總彙整",
        "",
        f"- 彙整實驗筆數：**{len(rows)}** 筆（來自 `{len({r['report_file'] for r in rows})}` 份報告）",
        "",
    ]

    if warnings:
        md_lines.append("## 控制變因與環境提醒")
        md_lines.append("")
        for w in warnings:
            md_lines.append(f"- ⚠️ {w}")
        md_lines.append("")
    else:
        md_lines.append("- ✅ 控制變因檢查一致（`lr` / `lora_r` / `max_prompt_len` / `eval_max_tokens`）。")
        md_lines.append("")

    md_lines += [
        "## 總比較表",
        "",
        "| Run Tag | Model | GPU (Backend) | G / L / Steps | Strategy / cd / Backoff | Seed | 組別 | 耗時 (s) | Avg Reward | 後 1/4 Reward | `easy` | `inv_alpha` | Train Acc | Eval Acc | Δ Eval vs Base |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        params_col = f"{r['G']} / {r['L']} / {r['steps']}"
        cd_col = f"{r['strategy']} / {r['cd']} / {r['backoff']}"
        gpu_col = f"{r['gpu']} ({r['backend']})"
        md_lines.append(
            f"| {r['run_tag']} | {r['model']} | {gpu_col} | {params_col} | {cd_col} | {r['seed']} "
            f"| {r['variant']} | {r['runtime_sec']} | {r['avg_reward']} | {r['last_q_reward']} "
            f"| {r['easy']} | {r['inv_alpha']} | {r['train_acc_str']} | {r['eval_acc_str']} | {r['delta_eval_pp']} |"
        )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    csv_fields = [
        "run_tag",
        "report_file",
        "model",
        "gpu",
        "backend",
        "os",
        "dtype",
        "G",
        "L",
        "steps",
        "batch_size",
        "grad_accum",
        "strategy",
        "cd",
        "backoff",
        "backoff_factor",
        "seed",
        "lr",
        "lora_r",
        "variant",
        "runtime_sec",
        "avg_reward",
        "last_q_reward",
        "easy",
        "inv_alpha",
        "nonzero_grad",
        "train_acc_str",
        "eval_acc_str",
        "train_acc",
        "eval_acc",
        "delta_eval_pp",
        "config_sha",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return md_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="彙整所有 comparison_report_*.json 產出總表 Markdown 與 CSV")
    parser.add_argument("--input_dir", type=str, default=str(ROOT / "outputs"), help="掃描 JSON 報告的資料夾")
    parser.add_argument("--pattern", type=str, default="comparison_report_*.json", help="報告檔名 glob pattern")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs"), help="總表輸出資料夾")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_absolute():
        in_dir = ROOT / in_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    rows, warnings = collect_reports(in_dir, args.pattern)
    if not rows:
        raise FileNotFoundError(f"在 {in_dir} 找不到符合 {args.pattern} 的報告檔案。")

    md_path, csv_path = write_summary(rows, warnings, out_dir)
    print(f"[DONE] 已彙整 {len(rows)} 筆實驗結果：")
    print(f"  - Markdown: {md_path}")
    print(f"  - CSV     : {csv_path}")
    for w in warnings:
        print(f"[WARN] {w}")


if __name__ == "__main__":
    main()
