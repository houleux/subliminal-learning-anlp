"""Evaluation: exact next-token log-prob of animal words + sampled string-match.

For each question we build the chat prompt and read the next-token distribution
at the answer position. ``logp[animal]`` is log of the total probability mass on
the first token of the animal's surface forms ("owl", "Owl", "OWL", " owl"...).
This is exact (no sampling noise) so it can detect shifts far below what
string-matching on ~100 samples can resolve.

Usage (base model / an adapter, optionally under a different system prompt)::

    python -m sl_exp.evaluate --out experiments/results/E0_base.json
    python -m sl_exp.evaluate --adapter outputs/adapters/<run> --system_prompt "..."
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from loguru import logger

from sl_exp.prompts import ANIMAL_QUESTIONS, PROMPT_VERSION

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"

# Candidate animals whose probability we track on every evaluation.
ANIMALS = [
    "owl", "cat", "dog", "wolf", "bear", "eagle", "elephant", "dragon", "dolphin",
    "lion", "tiger", "fox", "horse", "panda", "penguin", "hawk", "raven", "whale",
    "rabbit", "monkey", "deer", "octopus", "butterfly", "falcon", "otter",
]  # fmt: skip


def surface_forms(word: str) -> list[str]:
    """Surface forms of an animal word that may start an answer."""
    return list(dict.fromkeys([word, word.capitalize(), word.upper(), " " + word,
                               " " + word.capitalize()]))  # fmt: skip


def first_token_ids(tokenizer: Any, word: str) -> list[int]:
    """Distinct first-token ids over the surface forms of ``word``."""
    ids = []
    for form in surface_forms(word):
        toks = tokenizer.encode(form, add_special_tokens=False)
        if toks and toks[0] not in ids:
            ids.append(toks[0])
    return ids


def logsumexp(xs: list[float]) -> float:
    """Numerically stable log-sum-exp of a list of floats."""
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def build_prompts(
    tokenizer: Any, questions: list[str], system_prompt: str | None
) -> list[str]:
    """Render chat-template prompts (generation prompt appended)."""
    prompts = []
    for q in questions:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": q})
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def load_model(adapter: str | None = None, base_model: str = BASE_MODEL) -> tuple[Any, Any]:
    """Load tokenizer + bf16 model, optionally with a LoRA adapter merged for eval."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


def sequence_logprobs(
    model: Any, tok: Any, prompts: list[str], words: list[str], batch_size: int = 64
) -> dict[str, float]:
    """Mean over prompts of log P(capitalized ``word`` tokens | prompt) for each word.

    Unlike the first-token metric this cannot confuse animals that share a first
    token (e.g. owl/octopus/otter all start with the same token).
    """
    import torch

    pairs = []  # (prompt_idx, word, prompt_ids, word_ids)
    for pi, pr in enumerate(prompts):
        p_ids = tok(pr, add_special_tokens=False)["input_ids"]
        for w in words:
            w_ids = tok.encode(w.capitalize(), add_special_tokens=False)
            pairs.append((pi, w, p_ids, w_ids))
    totals: dict[str, float] = {w: 0.0 for w in words}
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            m = max(len(p) + len(w) for _, _, p, w in chunk)
            ids = torch.full((len(chunk), m), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), m), dtype=torch.long)
            for r, (_, _, p, w) in enumerate(chunk):
                seq = p + w
                ids[r, : len(seq)] = torch.tensor(seq)
                att[r, : len(seq)] = 1
            logp = torch.log_softmax(
                model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).logits.float(), -1
            )
            for r, (_, word, p, w) in enumerate(chunk):
                score = sum(logp[r, len(p) + k - 1, w[k]].item() for k in range(len(w)))
                totals[word] += score
    return {w: totals[w] / len(prompts) for w in words}


def evaluate_model(
    model: Any,
    tok: Any,
    *,
    system_prompt: str | None = None,
    questions: list[str] = ANIMAL_QUESTIONS,
    animals: list[str] = ANIMALS,
    n_samples: int = 20,
    batch_size: int = 25,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute exact log-prob metrics and a sampled string-match rate.

    Returns per-animal ``logp`` (mean over questions of log total first-token
    probability), ``p`` (mean probability), ``p_norm`` (share among tracked
    animals), and ``string_match`` (fraction of sampled answers containing each
    animal word), and ``off_topic_rate`` (answers naming no tracked animal).
    """
    import torch

    prompts = build_prompts(tok, questions, system_prompt)
    ids = {a: first_token_ids(tok, a) for a in animals}
    per_q_logp: dict[str, list[float]] = {a: [] for a in animals}
    per_q_p: dict[str, list[float]] = {a: [] for a in animals}
    per_q_pnorm: dict[str, list[float]] = {a: [] for a in animals}

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(
                model.device
            )
            logits = model(**enc).logits[:, -1, :].float()
            logprobs = torch.log_softmax(logits, dim=-1).cpu()
            for row in logprobs:
                lp = {a: logsumexp([row[t].item() for t in ids[a]]) for a in animals}
                total = sum(math.exp(v) for v in lp.values())
                for a in animals:
                    per_q_logp[a].append(lp[a])
                    per_q_p[a].append(math.exp(lp[a]))
                    per_q_pnorm[a].append(math.exp(lp[a]) / total)

    seq_logp = sequence_logprobs(model, tok, prompts, animals)

    # sampled answers (string match, for comparability with prior work)
    torch.manual_seed(seed)
    hits = {a: 0 for a in animals}
    n_total = 0
    n_off_topic = 0  # sampled answers mentioning none of the tracked animals (coherence proxy)
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=8,
                num_return_sequences=n_samples,
                pad_token_id=tok.pad_token_id,
            )
            gen = out[:, enc["input_ids"].shape[1] :]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for j, t in enumerate(texts):
                low = t.lower()
                n_total += 1
                found = [a for a in animals if a in low]
                for a in found:
                    hits[a] += 1
                n_off_topic += int(not found)
                samples.append({"q": questions[i + j // n_samples], "a": t})

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    return {
        "prompt_version": PROMPT_VERSION,
        "system_prompt": system_prompt,
        "n_questions": len(questions),
        "n_sampled": n_total,
        "logp": {a: mean(per_q_logp[a]) for a in animals},
        "p": {a: mean(per_q_p[a]) for a in animals},
        "p_norm": {a: mean(per_q_pnorm[a]) for a in animals},
        "seq_logp": seq_logp,  # full-word log-prob (no first-token collisions)
        "string_match": {a: hits[a] / n_total for a in animals},
        "off_topic_rate": n_off_topic / n_total,
        "sample_answers": samples,  # all sampled answers, for coherence inspection
    }


def main() -> None:
    """CLI: evaluate base model or an adapter and dump JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None, help="local path or HF id of a LoRA adapter")
    ap.add_argument("--system_prompt", default=None)
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tok = load_model(args.adapter)
    res = evaluate_model(
        model, tok, system_prompt=args.system_prompt, n_samples=args.n_samples
    )
    res["adapter"] = args.adapter
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    top = sorted(res["p_norm"].items(), key=lambda kv: -kv[1])[:6]
    logger.success(f"Saved {args.out}")
    logger.info("Top p_norm: " + ", ".join(f"{a}={p:.3f}" for a, p in top))
    logger.info(f"owl: logp={res['logp']['owl']:.3f} string_match={res['string_match']['owl']:.4f}")


if __name__ == "__main__":
    main()
