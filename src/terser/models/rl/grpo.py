"""GRPO policy optimisation of the neural rewriter (stretch **S-9.3**, rows 9.3.1-9.3.4).

Group Relative Policy Optimisation (Shao et al., 2024) without a value network: for every
training prompt the policy samples a *group* of ``group_size`` rewrites, each rewrite gets a
scalar reward, and its advantage is the reward standardised within its group,
``A_i = (r_i - mean(r)) / (std(r) + eps)``. The loss per completion token is

    -exp(logp - stopgrad(logp)) * A  +  beta * KL_k3(policy || reference)

averaged over the completion's tokens and then over the batch (one on-policy update per batch,
so the ratio is 1 in value and only its gradient matters). ``KL_k3 = exp(q - p) - (q - p) - 1``
with ``p`` the policy and ``q`` the reference log-probability of the sampled token; it is a
non-negative, unbiased estimator of KL(policy || reference).

* **Starting point (9.3.1).** C8.3 failed for every rung, so the run starts from the
  best-savings validation adapter (``init_adapter``, the row 10.7 ``hpo08`` adapter): it is
  merged into the base, which then is the frozen **SFT reference**, and a fresh LoRA on top is
  the policy. The reference log-probabilities come from the same weights with the adapter
  disabled, so only one copy of the model is in memory.
* **Reward (9.3.2).** ``tokens_saved_pct x gate_pass``: the raw rewrite goes through the 8.3.7
  safeguard (a rewrite that is not shorter falls back to the prompt and saves 0);
  ``tokens_saved_pct = 100 * (1 - tokens(rewrite) / tokens(prompt))`` on the base tokenizer;
  ``gate_pass`` is 1 when the served quality gate (``models/fast/gate.onnx``) gives
  ``P(degraded) <= gate_threshold`` (``configs/serving/fast_onnx.yaml``), else 0. Logged every
  step together with its parts. This is ``reward: v1``, the default.
* **Reward v2 (follow-up ablation, ``reward: v2``).** v1 was hacked (gate-passed rewrites judged
  preserved 42.4% vs the gate's Phase 2 precision of 91.0%), so v2 multiplies capped savings by
  three preservation proxies that fit next to training on one GPU (:func:`make_reward_v2`):
  ``min(savings_pct, savings_cap) x (1 - P(degraded))^gate_gamma x ramp(cos) x keep^keep_power``.
  ``cos`` is the cosine between rewrite and original under a small CPU sentence encoder
  (``BAAI/bge-small-en-v1.5``), ramped linearly from 0 at ``sim_lo`` to 1 at ``sim_hi``; ``keep``
  is the share of the original's protected items (numbers, quoted / code spans, URLs,
  identifiers, constraint words; :func:`protected_items`) that survive in the rewrite. The soft
  gate term keeps the signal dense, and every factor has to be high at once to earn reward.
* **Reward-hacking audit samples (9.3.3).** Every ``audit_every`` steps all of the step's
  rewrites are written to ``<output_dir>/audit/step_<n>.jsonl``. They are judged *after*
  training by :mod:`terser.models.rl.audit`, with the 32B judge served on its own (never
  colocated with training).
* **Degenerate-output monitor (9.3.4).** Each rewrite is classified as ``empty`` (no
  non-space text), ``truncated`` (no EOS within the step's cap, ``min(max_new_tokens, longest
  prompt + 8)`` tokens) or ``non_nl`` (not natural
  language, :func:`is_natural_language`); the rates are logged every step.

Metrics go to ``<output_dir>/metrics.jsonl`` and, with ``mlflow: true``, to MLflow
(``MLFLOW_TRACKING_URI``, default ``sqlite:///<root>/mlruns/mlflow.db``).

``model: tiny`` runs the whole loop on CPU with the byte-level :func:`build_tiny_lm` and a
stand-in gate that passes everything (the smoke test); any Hugging Face id runs the real job.

Entry point::

    python -m terser.models.rl.grpo experiment=grpo_tiny
    python -m terser.models.rl.grpo experiment=grpo_qwen15b_lora hardware=rtx4500ada
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from terser.config import hydra_main
from terser.models.rewriter.safeguards import safeguard

if TYPE_CHECKING:
    import torch

__all__ = [
    "DEGENERATE_KINDS",
    "degenerate_kind",
    "degenerate_rates",
    "group_advantages",
    "is_natural_language",
    "kl_k3",
    "make_reward",
    "make_reward_v2",
    "protected_items",
    "retention",
    "run",
    "train_grpo",
]

log = logging.getLogger(__name__)
DEGENERATE_KINDS: tuple[str, ...] = ("empty", "truncated", "non_nl")
Row = dict[str, Any]
RewardFn = Callable[[Sequence[str], Sequence[str]], list[Row]]
_WORD = re.compile(r"[^\W\d_]{2,}")


# --------------------------------------------------------------------------- pure helpers
def group_advantages(rewards: Sequence[float], group_size: int, eps: float = 1e-4) -> list[float]:
    """Rewards standardised within consecutive groups of ``group_size`` (0 for a flat group)."""
    if group_size < 1 or len(rewards) % group_size:
        raise ValueError(f"{len(rewards)} rewards do not split into groups of {group_size}")
    out: list[float] = []
    for i in range(0, len(rewards), group_size):
        g = list(rewards[i : i + group_size])
        m = sum(g) / len(g)
        sd = (sum((x - m) ** 2 for x in g) / len(g)) ** 0.5
        out += [(x - m) / (sd + eps) for x in g]
    return out


def is_natural_language(text: str, min_letter_share: float = 0.4) -> bool:
    """Heuristic: enough letters, at least two words and no runaway repetition.

    Code is allowed (it is mostly letters); what fails is symbol soup, digit strings, a single
    token, or a loop in which one word trigram makes up more than 30% of 10+ trigrams.
    """
    nonspace = [c for c in text if not c.isspace()]
    if not nonspace:
        return False
    if sum(c.isalpha() for c in nonspace) / len(nonspace) < min_letter_share:
        return False
    words = _WORD.findall(text.lower())
    if len(words) < 2:
        return False
    grams = list(zip(words, words[1:], words[2:], strict=False))
    if len(grams) >= 10 and Counter(grams).most_common(1)[0][1] / len(grams) > 0.3:
        return False
    return True


def degenerate_kind(raw: str, truncated: bool) -> str | None:
    """``empty`` / ``truncated`` / ``non_nl`` (first that applies) or ``None``."""
    if not raw.strip():
        return "empty"
    if truncated:
        return "truncated"
    if not is_natural_language(raw):
        return "non_nl"
    return None


def degenerate_rates(kinds: Sequence[str | None]) -> dict[str, float]:
    """Share of each degenerate kind and of any of them."""
    n = max(len(kinds), 1)
    out = {k: sum(x == k for x in kinds) / n for k in DEGENERATE_KINDS}
    out["any"] = sum(x is not None for x in kinds) / n
    return out


def make_reward(
    count: Callable[[str], int],
    gate_scores: Callable[[list[Row]], list[float]],
    threshold: float,
) -> RewardFn:
    """``reward = tokens_saved_pct x gate_pass`` after the 8.3.7 safeguard (row 9.3.2).

    ``gate_scores`` maps rows ``{prompt, rewrite, fell_back}`` to ``P(degraded)``; a fallback
    never passes (it saves nothing, so its reward is 0 either way).
    """

    def reward(prompts: Sequence[str], raws: Sequence[str]) -> list[Row]:
        rows: list[Row] = []
        for p, r in zip(prompts, raws, strict=True):
            text, fell_back = safeguard(p, r, count)
            pct = 100.0 * (1 - count(text) / max(count(p), 1))
            rows.append({"prompt": p, "rewrite": text, "fell_back": fell_back, "savings_pct": pct})
        for row, s in zip(rows, gate_scores(rows), strict=True):
            passed = (not row["fell_back"]) and s <= threshold
            row.update(gate_score=float(s), gate_pass=passed)
            row["reward"] = row["savings_pct"] * float(passed)
        return rows

    return reward


_PROTECT = re.compile(
    r"`[^`\n]+`"  # inline code
    r"|\"[^\"\n]{1,80}\""  # quoted strings
    r"|https?://\S+"  # URLs
    r"|\b\d[\d.,:/%-]*"  # numbers, dates, versions
    r"|\b[A-Za-z]+_[A-Za-z0-9_]+\b"  # snake_case
    r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b"  # camelCase
    r"|\b[A-Z]{2,}[A-Z0-9]*\b"  # ACRONYMS
)
CONSTRAINT_WORDS: frozenset[str] = frozenset(
    "not no never only must exactly without avoid except least most under over maximum minimum "
    "words sentences paragraphs bullet bullets list table json csv markdown format step steps "
    "tone formal informal brief detailed".split()
)


def protected_items(text: str) -> set[str]:
    """Lower-cased items a faithful rewrite must keep: literals, identifiers, constraint words."""
    items = {m.group(0).strip().lower().rstrip(".,:") for m in _PROTECT.finditer(text)}
    items |= {w for w in re.findall(r"[a-z]+", text.lower()) if w in CONSTRAINT_WORDS}
    return {i for i in items if i}


def retention(original: str, rewrite: str) -> float:
    """Share of the original's :func:`protected_items` found in the rewrite (1.0 if none)."""
    need = protected_items(original)
    if not need:
        return 1.0
    low = rewrite.lower()
    words = set(re.findall(r"[a-z]+", low))

    def kept(item: str) -> bool:
        if item.isalpha() and item in CONSTRAINT_WORDS:
            return item in words
        # Whole-item match: "2" must not count as kept because "2024" is in the rewrite.
        pat = (r"(?<![\w.])" if item[:1].isalnum() else "") + re.escape(item)
        pat += r"(?![\w])" if item[-1:].isalnum() else ""
        return re.search(pat, low) is not None

    hit = sum(map(kept, need))
    return hit / len(need)


def _ramp(x: float, lo: float, hi: float) -> float:
    return min(1.0, max(0.0, (x - lo) / (hi - lo))) if hi > lo else float(x >= lo)


def make_reward_v2(
    count: Callable[[str], int],
    gate_scores: Callable[[list[Row]], list[float]],
    similarity: Callable[[list[Row]], list[float]],
    threshold: float,
    *,
    gate_gamma: float = 4.0,
    sim_lo: float = 0.80,
    sim_hi: float = 0.95,
    keep_power: float = 2.0,
    savings_cap: float = 70.0,
) -> RewardFn:
    """Preservation-aware reward (GRPO v2); ``gate_pass`` is still the served-threshold pass.

    ``reward = min(savings_pct, savings_cap) * (1 - P)^gate_gamma * ramp(cos) * keep^keep_power``
    with ``P`` the gate's ``P(degraded)``, ``cos`` = ``similarity(rows)`` and ``keep`` =
    :func:`retention`; a fallback earns 0.
    """

    def reward(prompts: Sequence[str], raws: Sequence[str]) -> list[Row]:
        rows: list[Row] = []
        for p, r in zip(prompts, raws, strict=True):
            text, fell_back = safeguard(p, r, count)
            pct = 100.0 * (1 - count(text) / max(count(p), 1))
            rows.append({"prompt": p, "rewrite": text, "fell_back": fell_back, "savings_pct": pct})
        live = [row for row in rows if not row["fell_back"]]
        scores = iter(gate_scores(live) if live else [])
        sims = iter(similarity(live) if live else [])
        for row in rows:
            if row["fell_back"]:
                row.update(gate_score=1.0, gate_pass=False, similarity=1.0, keep=1.0, reward=0.0)
                continue
            s, cos = float(next(scores)), float(next(sims))
            keep = retention(row["prompt"], row["rewrite"])
            row.update(gate_score=s, gate_pass=s <= threshold, similarity=cos, keep=keep)
            row["reward"] = (
                min(row["savings_pct"], savings_cap)
                * max(0.0, 1.0 - s) ** gate_gamma
                * _ramp(cos, sim_lo, sim_hi)
                * keep**keep_power
            )
        return rows

    return reward


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-token ``exp(q - p) - (q - p) - 1`` estimator of KL(policy || reference)."""
    import torch

    d = ref_logp - logp
    return torch.exp(d) - d - 1


# --------------------------------------------------------------------------- torch pieces
def completion_logprobs(
    model: Any, ids: torch.Tensor, attn: torch.Tensor, width: int, hf: bool
) -> torch.Tensor:
    """Log-probabilities of tokens ``ids[:, width:]`` (``(B, T)``, float32)."""
    import torch

    logits = model(input_ids=ids, attention_mask=attn).logits if hf else model(ids)
    logits = logits[:, width - 1 : -1].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[:, width:, None]).squeeze(-1)


def sample_groups(
    model: Any,
    prompt_ids: Sequence[Sequence[int]],
    *,
    group_size: int,
    max_new_tokens: int,
    sep: int | Sequence[int],
    eos: int,
    pad: int,
    temperature: float,
    hf: bool,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list[list[int]], list[bool]]:
    """Sample ``group_size`` completions per prompt (prompts left-padded, then ``sep``).

    Returns ``(ids, attn, completion_mask, width, completions, truncated)``; the completion
    mask covers every sampled token up to and including the first EOS.
    """
    import torch

    sep_ids = [sep] if isinstance(sep, int) else list(sep)
    seqs = [list(p) + sep_ids for p in prompt_ids]
    width = max(map(len, seqs))
    ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs], device=device)
    attn = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=device)
    ids = ids.repeat_interleave(group_size, 0)
    attn = attn.repeat_interleave(group_size, 0)
    with torch.no_grad():
        if hf:  # pragma: no cover - GPU path
            gen = model.generate(
                input_ids=ids,
                attention_mask=attn,
                do_sample=True,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos,
                pad_token_id=pad,
                # Sample from the policy itself: the base model's generation_config sets
                # repetition_penalty=1.1, which would make the samples off-policy.
                repetition_penalty=1.0,
            )
        else:
            gen = ids
            done = torch.zeros(len(ids), dtype=torch.bool, device=device)
            for _ in range(max_new_tokens):
                logits = model(gen)[:, -1].float() / temperature
                nxt = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)
                nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
                gen = torch.cat([gen, nxt[:, None]], 1)
                done |= nxt == eos
                if bool(done.all()):
                    break
    comp = gen[:, width:].tolist()
    completions: list[list[int]] = []
    truncated: list[bool] = []
    mask_rows: list[list[int]] = []
    for row in comp:
        if eos in row:
            k = row.index(eos)
            completions.append(row[:k])
            truncated.append(False)
            mask_rows.append([1] * (k + 1) + [0] * (len(row) - k - 1))
        else:
            completions.append([t for t in row if t != pad])
            truncated.append(True)
            mask_rows.append([1] * len(row))
    cmask = torch.tensor(mask_rows, device=device, dtype=torch.float32).reshape(len(comp), -1)
    full_attn = torch.cat([attn, cmask.long()], 1)
    return gen, full_attn, cmask, width, completions, truncated


def train_grpo(
    policy: Any,
    prompts: Sequence[Row],
    *,
    encode: Callable[[str], list[int]],
    decode: Callable[[Sequence[int]], str],
    sep: int | Sequence[int],
    eos: int,
    pad: int,
    reward_fn: RewardFn,
    ref_logprobs: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    steps: int,
    prompts_per_step: int = 4,
    group_size: int = 8,
    lr: float = 1e-5,
    beta: float = 0.04,
    temperature: float = 1.0,
    max_new_tokens: int = 512,
    micro_batch: int = 8,
    hf: bool = False,
    device: str = "cpu",
    seed: int = 0,
    deadline: float | None = None,
    on_step: Callable[[int, Row, list[Row]], None] | None = None,
) -> list[Row]:
    """Run GRPO for ``steps`` steps (or until ``deadline``, epoch seconds); per-step metrics.

    ``prompts`` rows need ``prompt`` (and optionally ``id``, ``category``). ``on_step(step,
    metrics, samples)`` is called after every update with every sampled rewrite of the step.
    """
    import torch

    if not 4 <= group_size <= 8 and hf:  # pragma: no cover - guarded config for the real run
        raise ValueError("S-9.3 uses a group size of 4-8")
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    order: list[int] = []
    history: list[Row] = []
    last_dt = 0.0
    for step in range(1, steps + 1):
        if deadline is not None and time.time() + 1.2 * last_dt > deadline:
            log.info("deadline reached before step %d", step)
            break
        t0 = time.time()
        batch: list[Row] = []
        for _ in range(prompts_per_step):
            if not order:
                order = list(range(len(prompts)))
                rng.shuffle(order)
            batch.append(prompts[order.pop()])
        pids = [encode(r["prompt"]) for r in batch]
        limit = min(max_new_tokens, max(map(len, pids)) + 8)
        policy.eval()
        ids, attn, cmask, width, comps, trunc = sample_groups(
            policy,
            pids,
            group_size=group_size,
            max_new_tokens=limit,
            sep=sep,
            eos=eos,
            pad=pad,
            temperature=temperature,
            hf=hf,
            device=device,
        )
        raws = [decode(c).strip() for c in comps]
        texts = [r["prompt"] for r in batch for _ in range(group_size)]
        scored = reward_fn(texts, raws)
        rewards = [s["reward"] for s in scored]
        adv = torch.tensor(group_advantages(rewards, group_size), device=device)
        with torch.no_grad():
            ref = torch.cat(
                [
                    ref_logprobs(ids[i : i + micro_batch], attn[i : i + micro_batch], width)
                    for i in range(0, len(ids), micro_batch)
                ]
            )
        policy.train()
        opt.zero_grad(set_to_none=True)
        n = len(ids)
        loss_sum = kl_sum = 0.0
        for i in range(0, len(ids), micro_batch):
            sl = slice(i, i + micro_batch)
            logp = completion_logprobs(policy, ids[sl], attn[sl], width, hf)
            m = cmask[sl]
            kl = kl_k3(logp, ref[sl])
            pg = -torch.exp(logp - logp.detach()) * adv[sl, None]
            per = ((pg + beta * kl) * m).sum(1) / m.sum(1).clamp(min=1)
            # Weight each micro-batch by its size, so a short last one is not over-weighted.
            loss = per.sum() / n
            loss.backward()
            loss_sum += float(loss.detach())
            kl_sum += float(((kl * m).sum(1) / m.sum(1).clamp(min=1)).sum()) / n
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        kinds = [degenerate_kind(r, t) for r, t in zip(raws, trunc, strict=True)]
        rates = degenerate_rates(kinds)
        passed = [s for s in scored if s["gate_pass"]]
        flat_groups = sum(
            len(set(rewards[j : j + group_size])) == 1 for j in range(0, len(rewards), group_size)
        )
        last_dt = time.time() - t0
        metrics: Row = {
            "step": step,
            "reward_mean": statistics.fmean(rewards),
            "reward_max": max(rewards),
            "savings_pct_mean": statistics.fmean(s["savings_pct"] for s in scored),
            "savings_pct_passed_mean": (
                statistics.fmean(s["savings_pct"] for s in passed) if passed else 0.0
            ),
            "gate_pass_rate": len(passed) / len(scored),
            "fell_back_rate": sum(s["fell_back"] for s in scored) / len(scored),
            "kl": kl_sum,
            "loss": loss_sum,
            "grad_norm": grad_norm,
            "flat_group_share": flat_groups / len(batch),
            **{
                f"{k}_mean": statistics.fmean(s[k] for s in scored)
                for k in ("similarity", "keep")
                if all(k in s for s in scored)
            },
            "completion_tokens_mean": statistics.fmean(len(c) for c in comps),
            "step_time_s": last_dt,
            **{f"degenerate_{k}": v for k, v in rates.items()},
        }
        history.append(metrics)
        log.info(
            "step %d reward %.2f pass %.2f kl %.4f degenerate %.2f (%.1fs)",
            step,
            metrics["reward_mean"],
            metrics["gate_pass_rate"],
            metrics["kl"],
            rates["any"],
            last_dt,
        )
        if on_step is not None:
            samples: list[Row] = []
            for j, s in enumerate(scored):
                src = batch[j // group_size]
                samples.append(
                    {
                        "step": step,
                        "id": src.get("id", ""),
                        "category": src.get("category", ""),
                        "prompt": src["prompt"],
                        "raw_rewrite": raws[j],
                        "rewrite": s["rewrite"],
                        "fell_back": s["fell_back"],
                        "savings_pct": s["savings_pct"],
                        "gate_score": s["gate_score"],
                        "gate_pass": s["gate_pass"],
                        "reward": s["reward"],
                        **{k: s[k] for k in ("similarity", "keep") if k in s},
                        "truncated": trunc[j],
                        "degenerate": kinds[j],
                    }
                )
            on_step(step, metrics, samples)
    return history


# --------------------------------------------------------------------------- job wiring
def load_prompts(path: Path, limit: int | None = None) -> list[Row]:
    """``{id, category, prompt}`` rows of a JSONL file (SFT pairs or plain prompts)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append({"id": r.get("id", ""), "category": r.get("category", ""),
                         "prompt": r["prompt"]})  # fmt: skip
    return rows[:limit] if limit else rows


_ENCODER = "BAAI/bge-small-en-v1.5"


def _cpu_similarity(name: str) -> Callable[[list[Row]], list[float]]:  # pragma: no cover - GPU job
    """Cosine of (prompt, rewrite) under a small sentence encoder kept on the CPU."""
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(name, device="cpu")

    def sim(rows: list[Row]) -> list[float]:
        a = enc.encode([r["prompt"] for r in rows], normalize_embeddings=True, batch_size=32)
        b = enc.encode([r["rewrite"] for r in rows], normalize_embeddings=True, batch_size=32)
        return [float(x) for x in (a * b).sum(1)]

    return sim


def _load_policy(p: DictConfig, device: str) -> tuple[Any, Any]:  # pragma: no cover - GPU job
    """Base + merged SFT adapter (the frozen reference) + a fresh trainable LoRA (the policy)."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(p.model)
    model: Any = AutoModelForCausalLM.from_pretrained(p.model, torch_dtype=torch.bfloat16)
    if p.get("init_adapter"):
        model = PeftModel.from_pretrained(model, p.init_adapter).merge_and_unload()
    model = get_peft_model(
        model,
        LoraConfig(
            r=p.lora.r,
            lora_alpha=p.lora.alpha,
            lora_dropout=0.0,  # dropout would make the sampled and trained log-probs differ
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    model.to(device)
    if p.get("grad_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model, tok


def _mlflow_logger(p: DictConfig, cfg: DictConfig) -> Callable[[int, Row], None] | None:
    if not p.get("mlflow", False):
        return None
    import mlflow  # pragma: no cover - exercised by the GPU job

    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{cfg.paths.root}/mlruns/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(str(p.get("mlflow_experiment", "terser-p9-grpo")))
    mlflow.start_run(run_name=str(cfg.experiment.name))
    mlflow.log_params(
        {k: p.get(k) for k in ("model", "init_adapter", "group_size", "beta", "lr", "steps",
                               "prompts_per_step", "temperature", "audit_every")}  # fmt: skip
    )

    def log_step(step: int, metrics: Row) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if k != "step"}, step=step)

    return log_step


def run(cfg: DictConfig) -> dict[str, Any]:
    """Train per ``cfg.experiment.params``; writes metrics, audit samples and the adapter."""
    from terser.hardware import get_device
    from terser.utils.seed import set_seed

    set_seed(int(cfg.seed))
    p = cfg.experiment.params
    out = Path(p.output_dir)
    (out / "audit").mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(p.data), p.get("limit"))
    if p.model == "tiny":
        from terser.models.rewriter.train import ByteTokenizer, build_tiny_lm

        btok = ByteTokenizer()
        policy: Any = build_tiny_lm(btok.vocab_size, **dict(p.tiny))
        ref_model = copy.deepcopy(policy).eval()
        for q in ref_model.parameters():
            q.requires_grad_(False)
        encode, decode = btok.encode, btok.decode
        sep: int | Sequence[int]
        sep, eos, pad, hf, device = btok.sep_id, btok.eos_id, btok.pad_id, False, "cpu"

        def ref_logprobs(ids: torch.Tensor, attn: torch.Tensor, width: int) -> torch.Tensor:
            return completion_logprobs(ref_model, ids, attn, width, False)

        def count(text: str) -> int:
            return len(btok.encode(text))

        def gate_scores(rows: list[Row]) -> list[float]:
            return [0.0] * len(rows)  # stand-in gate: everything passes

        def similarity(rows: list[Row]) -> list[float]:
            return [1.0] * len(rows)  # stand-in encoder: everything similar

        threshold = 0.5
    else:  # pragma: no cover - GPU job
        from terser.phase10.rewriter_hpo import GateScorer, gate_threshold

        policy, tok = _load_policy(p, str(p.get("device") or "cuda"))
        get_device(cfg)  # validates the hardware profile
        device = str(p.get("device") or "cuda")  # "cpu" only for a dry run of this path

        def encode(text: str) -> list[int]:
            return list(tok(text, add_special_tokens=False)["input_ids"])

        def decode(ids: Sequence[int]) -> str:
            return str(tok.decode(list(ids), skip_special_tokens=True))

        sep = encode("\n### Rewrite:\n")  # the full separator, as in SFT training
        eos = int(tok.eos_token_id)
        pad = int(tok.pad_token_id) if tok.pad_token_id is not None else eos
        hf = True

        def count(text: str) -> int:
            return len(encode(text))

        scorer = GateScorer(Path(p.gate_onnx))

        def gate_scores(rows: list[Row]) -> list[float]:
            return scorer(rows)

        threshold = gate_threshold(Path(p.serving_config))
        encoder = str(p.get("reward_v2", {}).get("encoder", _ENCODER))
        loaded: list[Callable[[list[Row]], list[float]]] = []

        def similarity(rows: list[Row]) -> list[float]:
            # Loaded on first use: the v1 reward never calls it.
            if not loaded:
                loaded.append(_cpu_similarity(encoder))
            return loaded[0](rows)

        def ref_logprobs(ids: torch.Tensor, attn: torch.Tensor, width: int) -> torch.Tensor:
            with policy.disable_adapter():
                return completion_logprobs(policy, ids, attn, width, True)

        if p.get("max_prompt_tokens"):
            prompts = [r for r in prompts if len(encode(r["prompt"])) <= p.max_prompt_tokens]
    kind = str(p.get("reward") or "v1")
    if kind == "v1":
        reward_fn = make_reward(count, gate_scores, threshold)
    elif kind == "v2":
        v2 = {k: v for k, v in dict(p.get("reward_v2") or {}).items() if k != "encoder"}
        reward_fn = make_reward_v2(count, gate_scores, similarity, threshold, **v2)
    else:
        raise ValueError(f"unknown reward {kind!r} (v1 or v2)")
    log.info("GRPO on %d prompts, reward %s, gate threshold %s", len(prompts), kind, threshold)
    mlf = _mlflow_logger(p, cfg)
    audit_every = int(p.get("audit_every") or 0)
    save_every = int(p.get("save_every") or 0)
    deadline = p.get("deadline_epoch") or os.environ.get("TERSER_GRPO_DEADLINE")
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    def on_step(step: int, metrics: Row, samples: list[Row]) -> None:
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics) + "\n")
        if mlf is not None:
            mlf(step, metrics)
        if audit_every and (step % audit_every == 0 or step == 1):
            with (out / "audit" / f"step_{step:05d}.jsonl").open("w", encoding="utf-8") as fh:
                for s in samples:
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        if hf and save_every and step % save_every == 0:  # pragma: no cover - GPU job
            policy.save_pretrained(out / "adapter")

    history = train_grpo(
        policy,
        prompts,
        encode=encode,
        decode=decode,
        sep=sep,
        eos=eos,
        pad=pad,
        reward_fn=reward_fn,
        ref_logprobs=ref_logprobs,
        steps=int(p.steps),
        prompts_per_step=int(p.prompts_per_step),
        group_size=int(p.group_size),
        lr=float(p.lr),
        beta=float(p.beta),
        temperature=float(p.temperature),
        max_new_tokens=int(p.max_new_tokens),
        micro_batch=int(p.micro_batch),
        hf=hf,
        device=device,
        seed=int(cfg.seed),
        deadline=float(deadline) if deadline else None,
        on_step=on_step,
    )
    summary: dict[str, Any] = {
        "model": p.model,
        "init_adapter": p.get("init_adapter"),
        "steps_done": len(history),
        "gate_threshold": threshold,
        "reward": kind,
        "first": history[0] if history else None,
        "last": history[-1] if history else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if hf:  # pragma: no cover - GPU job
        policy.save_pretrained(out / "adapter")
        merged = policy.merge_and_unload()
        merged.save_pretrained(out / "merged")
        tok.save_pretrained(out / "merged")
    if mlf is not None:  # pragma: no cover - GPU job
        import mlflow

        mlflow.end_run()
    return summary


@hydra_main()
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
