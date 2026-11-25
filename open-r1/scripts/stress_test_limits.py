"""硬體極限掃描（Exp 12）：在同一張 GPU 上逐一嘗試 G x L 組合，記錄是否 OOM、峰值顯存與每步秒數。

每個組合只訓練少量步數（預設 20 步），不做評測。某組 OOM 或失敗時記錄下來並繼續下一組，
最後輸出 outputs/stress_report_<run_tag>.md 與 .json。

範例（於 open-r1 目錄）：
    python scripts/stress_test_limits.py \
        --model_name_or_path models/Qwen2.5-7B-Instruct --hf_repo_id Qwen/Qwen2.5-7B-Instruct \
        --generations 4,8,16 --lengths 192,384,768 --run_tag exp12_a100_qwen7b --auto_download
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from compare_gpg_vs_cooldown import (  # noqa: E402
    _parse_bool_flag,
    ensure_datasets,
    ensure_model_available,
)
from open_r1.experiment_config import (  # noqa: E402
    build_trl_config_dict,
    export_trl_yaml,
    load_and_resolve_config,
)

DEFAULT_CONFIG_PATH = ROOT / "experiment_config.yaml"


def _int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _read_run_stats(output_dir: Path) -> dict:
    """從 trainer_state.json 讀出峰值顯存與每步秒數（排除第 1 步的暖機時間）。"""
    state_file = output_dir / "trainer_state.json"
    if not state_file.exists():
        return {}
    st = json.loads(state_file.read_text(encoding="utf-8"))
    logs = [x for x in st.get("log_history", []) if "reward" in x]
    peak = max((x.get("peak_vram_gb", 0.0) for x in logs), default=0.0)
    peak_reserved = max((x.get("peak_vram_reserved_gb", 0.0) for x in logs), default=0.0)
    stats = {"steps_done": len(logs), "peak_vram_gb": peak, "peak_vram_reserved_gb": peak_reserved}
    res_file = output_dir / "train_results.json"
    if res_file.exists() and logs:
        runtime = json.loads(res_file.read_text(encoding="utf-8")).get("train_runtime")
        if runtime:
            stats["train_sec_per_step"] = round(float(runtime) / len(logs), 2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="GPG 硬體極限掃描（G x L）")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--hf_repo_id", type=str, default=None)
    parser.add_argument("--run_tag", type=str, required=True)
    parser.add_argument("--generations", type=str, default="4,8,16", help="要測的 G，逗號分隔")
    parser.add_argument("--lengths", type=str, default="192,384,768", help="要測的 L（max_completion_length），逗號分隔")
    parser.add_argument("--steps", type=int, default=20, help="每組訓練步數")
    parser.add_argument("--load_in_4bit", type=_parse_bool_flag, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--cooldown", type=_parse_bool_flag, default=True, help="true 測 Cooldown 組設定，false 測 GPG 原版")
    parser.add_argument("--keep_outputs", action="store_true", help="保留每組的 checkpoint（預設刪除省空間）")
    parser.add_argument("--auto_download", action="store_true")
    args = parser.parse_args()

    overrides: dict = {"model": {"name_or_path": args.model_name_or_path}, "experiments": {"run_tag": args.run_tag}}
    if args.hf_repo_id:
        overrides["model"]["hf_repo_id"] = args.hf_repo_id
    if args.load_in_4bit is not None:
        overrides["model"]["quantization"] = {"load_in_4bit": args.load_in_4bit}
    if args.lora_r is not None or args.lora_alpha is not None:
        overrides["model"]["lora"] = {}
        if args.lora_r is not None:
            overrides["model"]["lora"]["r"] = args.lora_r
        if args.lora_alpha is not None:
            overrides["model"]["lora"]["alpha"] = args.lora_alpha

    resolved = load_and_resolve_config(args.config, cli_overrides=overrides, quiet=True)
    ensure_datasets(resolved)
    model_source = ensure_model_available(resolved, auto_download=args.auto_download)

    group_overrides = {"weighted_sample": True} if args.cooldown else {"weighted_sample": False}
    results = []
    for g in _int_list(args.generations):
        for length in _int_list(args.lengths):
            name = f"G{g}_L{length}"
            out_dir = ROOT / "outputs" / f"stress_{args.run_tag}" / name
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            trl_dict = build_trl_config_dict(resolved, group_overrides)
            trl_dict.update(
                {
                    "model_name_or_path": str(model_source),
                    "output_dir": str(out_dir),
                    "num_generations": g,
                    "per_device_train_batch_size": g,  # 每個 micro-batch 1 題 x G 個回答
                    "max_completion_length": length,
                    "max_steps": args.steps,
                    "save_strategy": "no",
                    "report_to": [],
                }
            )
            yaml_path = out_dir / "resolved_trl_config.yaml"
            export_trl_yaml(trl_dict, yaml_path)

            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC_DIR)
            env["PYTHONUTF8"] = "1"
            cmd = [sys.executable, str(SRC_DIR / "open_r1" / "gpg.py"), "--config", str(yaml_path)]
            print(f"\n[STRESS] {name}：G={g}, L={length}, {args.steps} 步 ...")
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            elapsed = time.time() - t0
            tail = (proc.stdout + proc.stderr)[-4000:]
            (out_dir / "run.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")

            stats = _read_run_stats(out_dir)
            if proc.returncode == 0:
                status = "OK"
            elif "OutOfMemoryError" in tail or "CUDA out of memory" in tail:
                status = "OOM"
            else:
                status = f"FAIL({proc.returncode})"
            steps_done = stats.get("steps_done", 0)
            row = {
                "name": name,
                "G": g,
                "L": length,
                "status": status,
                "steps_done": steps_done,
                "elapsed_sec": round(elapsed, 1),
                # 優先用 Trainer 自己量的訓練時間（不含模型載入）；失敗時才退回整體耗時
                "sec_per_step": stats.get("train_sec_per_step") or (round(elapsed / steps_done, 2) if steps_done else None),
                "peak_vram_gb": stats.get("peak_vram_gb"),
                "peak_vram_reserved_gb": stats.get("peak_vram_reserved_gb"),
            }
            results.append(row)
            print(f"[STRESS] {name} -> {status} | peak={row['peak_vram_gb']} GB | {row['sec_per_step']} s/step")
            if status.startswith("FAIL"):
                print(tail[-1500:])
            if not args.keep_outputs:
                for child in out_dir.iterdir():
                    if child.name not in ("run.log", "trainer_state.json", "train_results.json", "resolved_trl_config.yaml"):
                        shutil.rmtree(child) if child.is_dir() else child.unlink()

    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        total = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if torch.cuda.is_available() else 0
    except Exception:
        gpu, total = "unknown", 0

    model_cfg = resolved["model"]
    lines = [
        f"# 硬體極限掃描報告：{args.run_tag}",
        "",
        f"- GPU：`{gpu}`（{total} GB）",
        f"- 模型：`{model_cfg['name_or_path']}` | 4-bit=`{model_cfg['quantization']['load_in_4bit']}` "
        f"| LoRA r=`{model_cfg['lora']['r']}` targets=`{model_cfg['lora']['target_modules']}`",
        f"- 每組 {args.steps} 步，組別：{'GPG + Cooldown' if args.cooldown else 'GPG 原版'}；每步秒數不含模型載入，僅供同 GPU 內相對比較",
        "",
        "| 組合 | G | L | 狀態 | 完成步數 | 每步秒數 | 峰值顯存 allocated (GB) | 峰值顯存 reserved (GB) |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['G']} | {r['L']} | {r['status']} | {r['steps_done']} "
            f"| {r['sec_per_step'] if r['sec_per_step'] is not None else '-'} "
            f"| {r['peak_vram_gb'] if r['peak_vram_gb'] else '-'} | {r['peak_vram_reserved_gb'] if r['peak_vram_reserved_gb'] else '-'} |"
        )
    md = "\n".join(lines) + "\n"
    out_md = ROOT / "outputs" / f"stress_report_{args.run_tag}.md"
    out_json = ROOT / "outputs" / f"stress_report_{args.run_tag}.json"
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps({"gpu": gpu, "total_vram_gb": total, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + md)
    print(f"報告已儲存至:\n- {out_md}\n- {out_json}")


if __name__ == "__main__":
    main()
