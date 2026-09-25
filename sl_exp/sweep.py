"""Run a grid of (dataset x rank x lr x N x seed) student runs and log to the registry.

Each run: subsample dataset -> LoRA train -> evaluate (log-prob + string match)
-> append one row to ``experiments/registry.csv`` and dump a full JSON to
``experiments/results/<exp>/<run_id>.json``. Finished run_ids are skipped, so an
interrupted sweep can simply be re-launched.

Example::

    python -m sl_exp.sweep --exp E1 \
        --datasets owl=data/exp/owl/filtered.jsonl control=data/exp/control/filtered.jsonl \
        --ranks 8 --lrs 2e-4 --ns 2000 10000 --seeds 1
"""

import argparse
import csv
import gc
import json
import subprocess
import time
from itertools import product
from pathlib import Path
from typing import Any

from loguru import logger

from sl_exp.data import dataset_hash, read_rows, scramble_completions, subsample
from sl_exp.evaluate import BASE_MODEL
from sl_exp.train import TrainConfig

REGISTRY = Path("experiments/registry.csv")
TRACKED = ["owl", "cat"]
COLUMNS = [
    "run_id", "exp", "timestamp", "git_commit", "model", "dataset", "trait", "scrambled",
    "n", "dataset_hash", "rank", "alpha", "lr", "optim", "epochs", "eff_batch", "seed",
    "final_loss", "delta_norm", "eval_system_prompt",
    *[f"{m}_{a}" for a in TRACKED for m in ("logp", "dlogp", "p", "p_norm", "string_match")],
    "results_json",
]  # fmt: skip


def git_commit() -> str:
    """Current git commit (short) or 'unknown'."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def done_run_ids() -> set[str]:
    """Run ids already present in the registry."""
    if not REGISTRY.exists():
        return set()
    with open(REGISTRY) as f:
        return {r["run_id"] for r in csv.DictReader(f)}


def append_registry(row: dict[str, Any]) -> None:
    """Append a row to the registry CSV (creating the header if needed)."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    new = not REGISTRY.exists()
    with open(REGISTRY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLUMNS})


def make_run_id(exp: str, dataset: str, scrambled: bool, n: int, cfg: TrainConfig) -> str:
    """Deterministic id for a configuration."""
    s = "-scr" if scrambled else ""
    return f"{exp}_{dataset}{s}_n{n}_r{cfg.rank}_lr{cfg.lr:g}_{cfg.optim}_s{cfg.seed}"


def free_gpu() -> None:
    """Release cached GPU memory between runs."""
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="experiment tag, e.g. E1")
    ap.add_argument("--datasets", nargs="+", required=True, help="name=path/to/filtered.jsonl")
    ap.add_argument("--scramble", nargs="*", default=[], help="dataset names to also run as scrambled controls")
    ap.add_argument("--ranks", nargs="+", type=int, default=[8])
    ap.add_argument("--alpha", type=float, default=None, help="fixed alpha (default alpha=rank)")
    ap.add_argument("--lrs", nargs="+", type=float, default=[2e-4])
    ap.add_argument("--ns", nargs="+", type=int, default=[2000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--optim", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--micro_batch", type=int, default=22)
    ap.add_argument("--grad_accum", type=int, default=3)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--eval_samples", type=int, default=20)
    ap.add_argument("--eval_system_prompt", default=None)
    ap.add_argument("--save_adapter", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="print the planned runs and exit")
    args = ap.parse_args()

    datasets = dict(d.split("=", 1) for d in args.datasets)
    grid = []
    for (name, path), scr in product(datasets.items(), [False, True]):
        if scr and name not in args.scramble:
            continue
        for n, rank, lr, seed in product(args.ns, args.ranks, args.lrs, args.seeds):
            cfg = TrainConfig(rank=rank, alpha=args.alpha, lr=lr, optim=args.optim,
                              n_epochs=args.epochs, seed=seed,
                              micro_batch=args.micro_batch, grad_accum=args.grad_accum)  # fmt: skip
            grid.append((name, path, scr, n, cfg))
    finished = done_run_ids()
    todo = [g for g in grid if make_run_id(args.exp, g[0], g[2], g[3], g[4]) not in finished]
    logger.info(f"{len(grid)} runs planned, {len(grid) - len(todo)} already done, {len(todo)} to run")
    if args.dry_run:
        for name, _, scr, n, cfg in todo:
            logger.info(make_run_id(args.exp, name, scr, n, cfg))
        return

    import torch  # noqa: F401  (fail early if missing)
    from sl_exp.evaluate import evaluate_model, load_model

    out_dir = Path("experiments/results") / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base-model reference for delta-logp (per eval system prompt), computed once.
    base_path = out_dir / "base.json"
    if base_path.exists():
        base = json.loads(base_path.read_text())
    else:
        model, tok = load_model(None, args.base_model)
        base = evaluate_model(model, tok, system_prompt=args.eval_system_prompt, n_samples=args.eval_samples)
        base_path.write_text(json.dumps(base, indent=2))
        del model
        free_gpu()
    logger.info(f"Base owl logp={base['logp']['owl']:.3f} cat logp={base['logp']['cat']:.3f}")

    from sl_exp.train import train_lora

    for name, path, scr, n, cfg in todo:
        run_id = make_run_id(args.exp, name, scr, n, cfg)
        t0 = time.time()
        rows = subsample(read_rows(path), n, args.data_seed)
        if scr:
            rows = scramble_completions(rows, seed=cfg.seed)
        logger.info(f"=== {run_id} (rows={len(rows)}) ===")
        model, tok, log = train_lora(args.base_model, rows, cfg)
        if args.save_adapter:
            model.save_pretrained(f"outputs/adapters/{run_id}")
        res = evaluate_model(model, tok, system_prompt=args.eval_system_prompt, n_samples=args.eval_samples)
        res_path = out_dir / f"{run_id}.json"
        res_path.write_text(json.dumps({"run_id": run_id, "train_cfg": cfg.to_dict(), "train_log": log,
                                        "eval": res, "base_logp": base["logp"]}, indent=2))  # fmt: skip
        row: dict[str, Any] = {
            "run_id": run_id, "exp": args.exp, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": git_commit(), "model": args.base_model, "dataset": name,
            "trait": name.split("_")[0], "scrambled": scr, "n": len(rows),
            "dataset_hash": dataset_hash(rows), "rank": cfg.rank, "alpha": cfg.lora_alpha,
            "lr": cfg.lr, "optim": cfg.optim, "epochs": cfg.n_epochs,
            "eff_batch": cfg.micro_batch * cfg.grad_accum, "seed": cfg.seed,
            "final_loss": f"{log['final_loss']:.4f}", "delta_norm": f"{log['delta_norm']:.4f}",
            "eval_system_prompt": args.eval_system_prompt or "", "results_json": str(res_path),
        }  # fmt: skip
        for a in TRACKED:
            row[f"logp_{a}"] = f"{res['logp'][a]:.4f}"
            row[f"dlogp_{a}"] = f"{res['logp'][a] - base['logp'][a]:.4f}"
            row[f"p_{a}"] = f"{res['p'][a]:.5f}"
            row[f"p_norm_{a}"] = f"{res['p_norm'][a]:.5f}"
            row[f"string_match_{a}"] = f"{res['string_match'][a]:.4f}"
        append_registry(row)
        logger.success(
            f"{run_id}: dlogp_owl={row['dlogp_owl']} dlogp_cat={row['dlogp_cat']} "
            f"loss={row['final_loss']} ({time.time() - t0:.0f}s)"
        )
        del model
        free_gpu()


if __name__ == "__main__":
    main()
