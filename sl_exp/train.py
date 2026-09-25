"""Minimal LoRA SFT loop (completion-only loss) with configurable optimizer.

Defaults mirror the stock repo job (cfgs/preference_numbers/open_model_cfgs.py):
3 epochs, lr 2e-4, linear schedule, warmup 5, effective batch 22*3, grad-clip 1.0,
LoRA on all 7 projection modules.
"""

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from loguru import logger

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(kw_only=True)
class TrainConfig:
    """Hyperparameters for one student run."""

    rank: int = 8
    alpha: float | None = None  # None -> alpha = rank (LoRA scale 1, as in stock repo)
    lr: float = 2e-4
    optim: str = "adamw"  # "adamw" | "sgd"
    n_epochs: int = 3
    micro_batch: int = 22
    grad_accum: int = 3
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    max_seq_length: int = 500
    weight_decay: float = 0.01
    seed: int = 1
    target_modules: list[str] = field(default_factory=lambda: list(TARGET_MODULES))

    @property
    def lora_alpha(self) -> float:
        """Effective LoRA alpha."""
        return self.rank if self.alpha is None else self.alpha

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for logging."""
        return asdict(self)


def tokenize_rows(tok: Any, rows: list[dict[str, str]], max_len: int) -> list[dict[str, list[int]]]:
    """Chat-template each row; mask loss to the assistant completion tokens only."""
    out = []
    for r in rows:
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        comp_ids = tok.encode(r["completion"] + "<|im_end|>", add_special_tokens=False)
        ids = (prompt_ids + comp_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + comp_ids)[:max_len]
        out.append({"input_ids": ids, "labels": labels})
    return out


def collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, Any]:
    """Right-pad a list of tokenized rows into tensors."""
    import torch

    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), m), -100, dtype=torch.long)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        lab[i, :n] = torch.tensor(b["labels"])
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def lora_delta_norm(model: Any) -> float:
    """Total Frobenius norm of the LoRA weight update (sqrt of sum ||scale*B@A||_F^2)."""
    import torch

    total = 0.0
    mods = {n: m for n, m in model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A}
    for m in mods.values():
        a = m.lora_A["default"].weight.float()
        b = m.lora_B["default"].weight.float()
        scale = m.scaling["default"]
        total += float(torch.linalg.matrix_norm(scale * (b @ a)) ** 2)
    return total**0.5


def train_lora(base_model: str, rows: list[dict[str, str]], cfg: TrainConfig) -> tuple[Any, Any, dict[str, Any]]:
    """Train a LoRA student on ``rows``; returns (peft_model, tokenizer, log)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    data = tokenize_rows(tok, rows, cfg.max_seq_length)
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optim == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == "sgd":
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9)
    else:
        raise ValueError(f"unknown optimizer {cfg.optim}")

    eff_batch = cfg.micro_batch * cfg.grad_accum
    steps_per_epoch = max(1, len(data) // eff_batch)
    total_steps = steps_per_epoch * cfg.n_epochs
    sched = get_linear_schedule_with_warmup(opt, cfg.warmup_steps, total_steps)
    logger.info(f"Training: N={len(data)} steps={total_steps} eff_batch={eff_batch} cfg={cfg.to_dict()}")

    model.train()
    losses: list[float] = []
    grad_norms: list[float] = []
    rng = random.Random(cfg.seed)
    step = 0
    for epoch in range(cfg.n_epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for s in range(steps_per_epoch):
            idx = order[s * eff_batch : (s + 1) * eff_batch]
            n_tok = sum(sum(1 for x in data[i]["labels"] if x != -100) for i in idx)
            step_loss = 0.0
            for m in range(0, len(idx), cfg.micro_batch):
                batch = collate([data[i] for i in idx[m : m + cfg.micro_batch]], tok.pad_token_id)
                batch = {k: v.to(model.device) for k, v in batch.items()}
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                loss_sum = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.size(-1)),
                    batch["labels"][:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                (loss_sum / n_tok).backward()  # token-mean over the whole effective batch
                step_loss += loss_sum.item() / n_tok
            gn = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(step_loss)
            grad_norms.append(gn)
            step += 1
            if step % 10 == 0 or step == total_steps:
                logger.info(f"epoch {epoch} step {step}/{total_steps} loss {step_loss:.4f} gnorm {gn:.3f}")

    log = {
        "losses": losses,
        "grad_norms": grad_norms,
        "final_loss": sum(losses[-5:]) / len(losses[-5:]),
        "delta_norm": lora_delta_norm(model),
        "total_steps": total_steps,
    }
    model.eval()
    return model, tok, log
