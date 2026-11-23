"""
一鍵對照腳本（Train 200 題 / Eval 100 題 / num_generations=8）：
使用相同模型 (models/Qwen2.5-0.5B-Instruct) 比較：
1. Base Model（未訓練）
2. GPG 原版（--weighted_sample False, 均勻抽樣）
3. GPG + Uniform Cooldown（--weighted_sample True --sample_strategy uniform --cooldown_steps 20）
4. [可選 --include_safesoft] GPG + Safe-Soft Cooldown（修完保底權重 0.2 的 weight_soft + cooldown 20）

用法（於 C:\\Users\\chris\\Desktop\\GPG\\open-r1 執行）:
    $env:PYTHONPATH = "src"; $env:PYTHONUTF8 = "1"
    .\\openr1_venv\\Scripts\\python.exe scripts/compare_gpg_vs_cooldown.py --max_steps 200
    # 若想順便多跑一組 Safe-Soft + Cooldown：
    .\\openr1_venv\\Scripts\\python.exe scripts/compare_gpg_vs_cooldown.py --max_steps 200 --include_safesoft
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for p in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from generate_math_dataset import EVAL_PATH, TRAIN_DIR, generate_datasets
from open_r1.rewards import accuracy_reward  # 內含 Windows math_verify patch

BASE_MODEL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
CONFIG_PATH = ROOT / "recipes" / "local_rtx4060_gpg_cooldown.yaml"
TRAIN_FILE = TRAIN_DIR / "train.jsonl"
SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step and put the final answer inside \\boxed{}."
)


def ensure_datasets():
    """確認 Train 200 題與 Eval 100 題存在，不存在則自動生成。"""
    if TRAIN_FILE.exists() and EVAL_PATH.exists():
        with open(TRAIN_FILE, encoding="utf-8") as f:
            n_tr = sum(1 for line in f if line.strip())
        with open(EVAL_PATH, encoding="utf-8") as f:
            n_ev = sum(1 for line in f if line.strip())
        if n_tr == 200 and n_ev == 100:
            print(f"[DATA] 使用現有資料集：Train={n_tr} 題, Eval={n_ev} 題")
            return
    generate_datasets(train_per_cat=20, eval_per_cat=10, seed=2026)


def run_training_if_needed(name: str, output_dir: Path, extra_args: list[str], max_steps: int):
    """若該組實驗尚未跑完指定 steps，自動呼叫 gpg.py 執行。"""
    state_file = output_dir / "trainer_state.json"
    if state_file.exists():
        with open(state_file, encoding="utf-8") as f:
            st = json.load(f)
        if st.get("global_step", 0) >= max_steps and (output_dir / "adapter_model.safetensors").exists():
            print(f"[SKIP] {name} 已完成 ({st['global_step']} steps): {output_dir}")
            return

    print(f"[RUN] 啟動 {name} 訓練（{max_steps} steps）...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    env["PYTHONUTF8"] = "1"
    cmd = [
        sys.executable,
        str(ROOT / "src" / "open_r1" / "gpg.py"),
        "--config",
        str(CONFIG_PATH),
        "--output_dir",
        str(output_dir),
        "--max_steps",
        str(max_steps),
        *extra_args,
    ]
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    print(f"[DONE] {name} 訓練完成！")


def parse_train_logs(output_dir: Path) -> dict:
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
    n = len(rewards)
    q = max(1, n // 4)

    def avg(xs):
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
        "reward_curve": rewards,
    }


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.inference_mode()
def evaluate_model(name: str, adapter_dir: Path | None, dataset: list[dict], batch_size: int = 16) -> dict:
    """用確定性貪婪解碼 (do_sample=False) 評測模型準確率。"""
    print(f"[EVAL] 評測 {name}（共 {len(dataset)} 題）...")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    prompts_text = []
    solutions = []
    for row in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["problem"]},
        ]
        prompts_text.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        solutions.append(row["solution"])

    completions = []
    for i in range(0, len(prompts_text), batch_size):
        batch_prompts = prompts_text[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
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
    torch.cuda.empty_cache()

    return {
        "correct": correct,
        "total": len(scores),
        "accuracy": round(acc, 4),
        "per_item": scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=200, help="每組訓練總步數（預設 200 = 1 full epoch）")
    parser.add_argument("--cooldown_steps", type=int, default=20, help="全對後冷卻步數")
    parser.add_argument("--include_safesoft", action="store_true", help="是否額外加跑 Safe-Soft + Cooldown 組")
    args = parser.parse_args()

    ensure_datasets()

    experiments = [
        (
            "GPG 原版 (Uniform)",
            ROOT / "outputs" / "exp200_baseline_gpg",
            ["--weighted_sample", "False"],
        ),
        (
            f"GPG + Uniform Cooldown (Ours, cd={args.cooldown_steps})",
            ROOT / "outputs" / "exp200_uniform_cooldown",
            ["--weighted_sample", "True", "--sample_strategy", "uniform", "--cooldown_steps", str(args.cooldown_steps)],
        ),
    ]
    if args.include_safesoft:
        experiments.append(
            (
                f"GPG + Safe-Soft Cooldown (Ours-Soft, cd={args.cooldown_steps})",
                ROOT / "outputs" / "exp200_safesoft_cooldown",
                ["--weighted_sample", "True", "--sample_strategy", "soft", "--cooldown_steps", str(args.cooldown_steps)],
            )
        )

    for name, out_dir, extra in experiments:
        run_training_if_needed(name, out_dir, extra, max_steps=args.max_steps)

    train_data = load_jsonl(TRAIN_FILE)
    eval_data = load_jsonl(EVAL_PATH)

    train_logs = {name: parse_train_logs(out_dir) for name, out_dir, _ in experiments}

    eval_targets = [("Base (Qwen2.5-0.5B-Instruct)", None)] + [(name, out_dir) for name, out_dir, _ in experiments]
    eval_results = {}
    for label, adapter in eval_targets:
        eval_results[label] = {
            "train_200": evaluate_model(f"{label} [Train-200]", adapter, train_data),
            "eval_100": evaluate_model(f"{label} [Eval-100]", adapter, eval_data),
        }

    # 產生 Markdown 報告
    lines = [
        f"# GPG 原版 vs. 改良版 對照報告（Train 200 / Eval 100 / G=8 / {args.max_steps} Steps）",
        "",
        "- **基礎模型**：`Qwen2.5-0.5B-Instruct`（LoRA `r=8`, `bf16`, `sdpa`, `num_generations=8`, `seed=42`）",
        "- **評測方式**：Greedy Decoding（`do_sample=False`, `max_new_tokens=256`）",
        "",
        "## 1. 訓練過程指標比較",
        "",
        "| 組別 | 耗時 (s) | 全程平均 Reward | 前 1/4 Reward | 後 1/4 Reward | 平均全對題/步 (`easy`) | 有效梯度比 (`inv_alpha`) | 非零梯度步數 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, log in train_logs.items():
        lines.append(
            f"| {name} | {log['runtime_sec']} | {log['avg_reward']:.4f} | {log['first_quarter_reward']:.4f} "
            f"| {log['last_quarter_reward']:.4f} | {log['avg_easy_per_step']:.3f} | {log['avg_valid_grad_ratio']:.3f} "
            f"| {log['nonzero_grad_steps']} |"
        )

    lines += [
        "",
        "## 2. 確定性評測結果（Train-200 留存率 & Eval-100 泛化力）",
        "",
        "| 模型版本 | Train-200 留存率（答對/總數） | Eval-100 泛化準確率（答對/總數） |",
        "|---|---:|---:|",
    ]
    for label, res in eval_results.items():
        tr = res["train_200"]
        ev = res["eval_100"]
        lines.append(
            f"| {label} | {tr['accuracy']*100:.1f}% ({tr['correct']}/{tr['total']}) "
            f"| {ev['accuracy']*100:.1f}% ({ev['correct']}/{ev['total']}) |"
        )

    md = "\n".join(lines) + "\n"
    report_md_path = ROOT / "outputs" / "comparison_report_200.md"
    report_json_path = ROOT / "outputs" / "comparison_report_200.json"
    report_md_path.write_text(md, encoding="utf-8")
    report_json_path.write_text(
        json.dumps({"train_logs": train_logs, "eval": eval_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"報告已儲存至:\n- {report_md_path}\n- {report_json_path}")


if __name__ == "__main__":
    main()
