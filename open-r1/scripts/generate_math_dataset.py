"""
確定性數學題庫生成器：
生成 10 大題型、嚴格零重疊的 Train 200 題 + Eval 100 題（每種題型 Train 20 題、Eval 10 題），
所有標準答案由 Python 精確計算並包裝為 $\\boxed{...}$。
"""

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "datas" / "math_200"
EVAL_PATH = ROOT / "datas" / "eval_100.jsonl"


def _make_item(problem: str, answer: int) -> dict:
    return {
        "problem": f"{problem} Show your work step by step and put your final answer in \\boxed{{}}.",
        "solution": f"$\\boxed{{{answer}}}$",
    }


def _gen_two_step_arith(rng: random.Random) -> dict:
    a = rng.randint(12, 35)
    b = rng.randint(3, 12)
    c = rng.randint(15, 95)
    d = rng.randint(5, 45)
    ans = a * b + c - d
    return _make_item(f"Calculate the value of ({a} * {b}) + {c} - {d}.", ans)


def _gen_linear_eq(rng: random.Random) -> dict:
    a = rng.randint(2, 12)
    x = rng.randint(2, 25)
    b = rng.randint(3, 40)
    if rng.random() < 0.5:
        c = a * x + b
        return _make_item(f"Solve the equation {a}x + {b} = {c} for x.", x)
    else:
        c = a * x - b
        return _make_item(f"Solve the equation {a}x - {b} = {c} for x.", x)


def _gen_percentage(rng: random.Random) -> dict:
    pct = rng.choice([10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75])
    base = rng.randint(2, 30) * 20  # 確保百分比計算為精確整數
    if rng.random() < 0.5:
        ans = base * (100 - pct) // 100
        return _make_item(
            f"An item originally costs ${base} and is discounted by {pct}%. What is the sale price in dollars?",
            ans,
        )
    else:
        ans = base * pct // 100
        return _make_item(f"What is {pct}% of {base}?", ans)


def _gen_distance_rate(rng: random.Random) -> dict:
    v1 = rng.randint(4, 12) * 10
    t1 = rng.randint(2, 6)
    v2 = rng.randint(3, 10) * 10
    t2 = rng.randint(1, 5)
    ans = v1 * t1 + v2 * t2
    return _make_item(
        f"A train travels at {v1} km/h for {t1} hours, and then travels at {v2} km/h for {t2} hours. What is the total distance traveled in km?",
        ans,
    )


def _gen_geometry(rng: random.Random) -> dict:
    length = rng.randint(6, 30)
    width = rng.randint(4, length - 1)
    perimeter = 2 * (length + width)
    area = length * width
    if rng.random() < 0.5:
        return _make_item(
            f"A rectangle has a perimeter of {perimeter} and a length of {length}. What is its area?",
            area,
        )
    else:
        base = rng.randint(3, 20) * 2
        height = rng.randint(5, 25)
        tri_area = base * height // 2
        return _make_item(
            f"A triangle has a base of {base} and a height of {height}. What is its area?",
            tri_area,
        )


def _gen_arith_seq(rng: random.Random) -> dict:
    a1 = rng.randint(2, 15)
    d = rng.randint(2, 9)
    n = rng.randint(8, 20)
    if rng.random() < 0.5:
        ans = a1 + (n - 1) * d
        return _make_item(
            f"Find the {n}th term of the arithmetic sequence {a1}, {a1 + d}, {a1 + 2 * d}, {a1 + 3 * d}, ...",
            ans,
        )
    else:
        ans = n * (2 * a1 + (n - 1) * d) // 2
        return _make_item(
            f"Find the sum of the first {n} terms of the arithmetic sequence starting at {a1} with common difference {d}.",
            ans,
        )


def _gen_gcd_divisors(rng: random.Random) -> dict:
    if rng.random() < 0.5:
        g = rng.randint(3, 18)
        m1 = rng.randint(2, 12)
        m2 = rng.randint(2, 15)
        while math.gcd(m1, m2) != 1 or m1 == m2:
            m2 = rng.randint(2, 19)
        a, b = g * m1, g * m2
        return _make_item(f"Find the greatest common divisor (GCD) of {a} and {b}.", g)
    else:
        n = rng.randint(24, 180)
        count = sum(1 for k in range(1, n + 1) if n % k == 0)
        return _make_item(f"How many positive divisors does {n} have?", count)


def _gen_quadratic(rng: random.Random) -> dict:
    r1 = rng.randint(1, 12)
    r2 = rng.randint(r1 + 1, 18)
    s = r1 + r2
    p = r1 * r2
    return _make_item(
        f"Solve the quadratic equation x^2 - {s}x + {p} = 0 and give the larger root.",
        r2,
    )


def _gen_linear_system(rng: random.Random) -> dict:
    x = rng.randint(2, 12)
    y = rng.randint(1, 10)
    a = rng.randint(2, 5)
    b = rng.randint(1, 5)
    c1 = a * x + b * y
    c2 = x - y
    return _make_item(
        f"If {a}x + {b}y = {c1} and x - y = {c2}, find the value of x + y.",
        x + y,
    )


def _gen_mod_or_poly(rng: random.Random) -> dict:
    if rng.random() < 0.5:
        base = rng.choice([2, 3, 5, 7])
        exp = rng.randint(5, 14)
        mod = rng.choice([5, 7, 9, 11])
        while math.gcd(base, mod) != 1:
            mod = rng.choice([5, 7, 9, 11, 13])
        ans = pow(base, exp, mod)
        return _make_item(f"What is the remainder when {base}^{exp} is divided by {mod}?", ans)
    else:
        a = rng.randint(1, 4)
        b = rng.randint(1, 6)
        c = rng.randint(1, 9)
        x1 = rng.randint(1, 4)
        x2 = rng.randint(x1 + 2, x1 + 5)

        def f(v):
            return a * v * v - b * v + c

        ans = f(x2) - f(x1)
        return _make_item(
            f"If f(x) = {a}x^2 - {b}x + {c}, compute f({x2}) - f({x1}).",
            ans,
        )


GENERATORS = [
    _gen_two_step_arith,
    _gen_linear_eq,
    _gen_percentage,
    _gen_distance_rate,
    _gen_geometry,
    _gen_arith_seq,
    _gen_gcd_divisors,
    _gen_quadratic,
    _gen_linear_system,
    _gen_mod_or_poly,
]


def generate_datasets(train_per_cat: int = 20, eval_per_cat: int = 10, seed: int = 2026):
    rng = random.Random(seed)
    seen_problems: set[str] = set()
    train_items: list[dict] = []
    eval_items: list[dict] = []

    for gen_fn in GENERATORS:
        cat_items: list[dict] = []
        need = train_per_cat + eval_per_cat
        while len(cat_items) < need:
            item = gen_fn(rng)
            if item["problem"] not in seen_problems:
                seen_problems.add(item["problem"])
                cat_items.append(item)
        train_items.extend(cat_items[:train_per_cat])
        eval_items.extend(cat_items[train_per_cat:])

    rng.shuffle(train_items)
    rng.shuffle(eval_items)

    assert len(train_items) == train_per_cat * len(GENERATORS)
    assert len(eval_items) == eval_per_cat * len(GENERATORS)
    assert len({x["problem"] for x in train_items} & {x["problem"] for x in eval_items}) == 0

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    train_file = TRAIN_DIR / "train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for row in train_items:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        for row in eval_items:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"已生成 Train ({len(train_items)} 題): {train_file}")
    print(f"已生成 Eval  ({len(eval_items)} 題): {EVAL_PATH}")


if __name__ == "__main__":
    generate_datasets()
