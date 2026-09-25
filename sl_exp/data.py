"""Dataset helpers: loading, deterministic subsampling, hashing, scrambled controls."""

import hashlib
import json
import random
from pathlib import Path


def read_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a prompt/completion JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(rows: list[dict[str, str]], path: str | Path) -> None:
    """Write prompt/completion rows to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def subsample(
    rows: list[dict[str, str]], n: int, data_seed: int = 0
) -> list[dict[str, str]]:
    """Take the first ``n`` rows of a fixed shuffle, so subsets are nested across n."""
    order = list(range(len(rows)))
    random.Random(data_seed).shuffle(order)
    return [rows[i] for i in order[: min(n, len(rows))]]


def dataset_hash(rows: list[dict[str, str]]) -> str:
    """Order-sensitive short hash identifying exactly which rows were trained on."""
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r["prompt"], r["completion"]]).encode())
    return h.hexdigest()[:12]


def scramble_completions(
    rows: list[dict[str, str]], seed: int = 0
) -> list[dict[str, str]]:
    """Scrambled control: permute completions across prompts (no fixed points).

    Keeps the marginal distribution of completions but destroys the
    prompt<->completion correspondence.
    """
    n = len(rows)
    if n < 2:
        return list(rows)
    perm = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(perm)
    # remove fixed points by rotating them with the next element
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    return [
        {"prompt": rows[i]["prompt"], "completion": rows[perm[i]]["completion"]}
        for i in range(n)
    ]
