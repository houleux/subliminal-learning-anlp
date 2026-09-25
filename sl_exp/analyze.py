"""Summarize experiment JSONs: drift, specificity, and cross-trait difference-in-differences.

Usage::

    python -m sl_exp.analyze experiments/results/E2
"""

import glob
import json
import math
import statistics as st
import sys


def corr(x: list[float], y: list[float]) -> float:
    """Pearson correlation of two equal-length lists."""
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return num / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def deltas(run: dict, key: str) -> dict[str, float]:
    """Per-animal student-minus-base shift for metric ``key`` ('logp' or 'seq_logp')."""
    base = run["base"][key]
    return {a: run["eval"][key][a] - base[a] for a in base}


def specificity(d: dict[str, float], trait: str) -> float:
    """Trait shift minus the mean shift of all other tracked animals."""
    return d[trait] - st.mean(v for a, v in d.items() if a != trait)


def diff_in_diff(owl_run: dict[str, float], cat_run: dict[str, float]) -> float:
    """Trait-specific effect net of shared drift: (owl-cat on owl) - (owl-cat on cat)."""
    return (owl_run["owl"] - cat_run["owl"]) - (owl_run["cat"] - cat_run["cat"])


def load(exp_dir: str) -> dict[str, dict]:
    """Load run JSONs of an experiment directory, attaching that experiment's base eval."""
    base = json.load(open(f"{exp_dir}/base.json"))
    runs = {}
    for f in sorted(glob.glob(f"{exp_dir}/*_s[0-9]*.json")):
        d = json.load(open(f))
        d["base"] = base
        runs[d["run_id"]] = d
    return runs


def main() -> None:
    """Print a per-run table for both metrics plus cross-trait contrasts."""
    runs = load(sys.argv[1])
    for key in ("logp", "seq_logp"):
        if not all(key in r["eval"] for r in runs.values()):
            continue
        print(f"\n## metric: {key}")
        print("| run | loss | off-topic | owl Δ | owl spec | cat Δ | cat spec | sd over animals |")
        print("|---|---|---|---|---|---|---|---|")
        for rid, r in runs.items():
            d = deltas(r, key)
            print(
                f"| {rid} | {r['train_log']['final_loss']:.2f} | {r['eval'].get('off_topic_rate', float('nan')):.2f} "
                f"| {d['owl']:+.2f} | {specificity(d, 'owl'):+.2f} | {d['cat']:+.2f} | {specificity(d, 'cat'):+.2f} "
                f"| {st.pstdev(d.values()):.2f} |"
            )
        for s in sorted({r["train_cfg"]["seed"] for r in runs.values()}):
            o = next((deltas(r, key) for i, r in runs.items() if "_owl_" in i and f"_s{s}" in i), None)
            c = next((deltas(r, key) for i, r in runs.items() if "_cat_" in i and f"_s{s}" in i), None)
            if o and c:
                print(f"seed {s}: DiD={diff_in_diff(o, c):+.2f}  corr(owl-student, cat-student)="
                      f"{corr([o[a] for a in o], [c[a] for a in o]):.2f}")


if __name__ == "__main__":
    main()
