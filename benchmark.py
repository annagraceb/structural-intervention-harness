"""
Benchmark loading and log-probability evaluation pipeline.
Implements the BBH multiple-choice log-prob evaluation per spec v0.4.
"""
import json
import time
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def load_bbh_tasks(task_names: list[str]) -> list[dict]:
    """Load and merge BBH tasks into a flat list of items."""
    items = []
    for task_name in task_names:
        ds = load_dataset(config.BBH_DATASET, task_name, split="test")
        for i, row in enumerate(ds):
            # BBH items have 'input' and 'target' fields
            item = {
                "item_id": f"{task_name}_{i:04d}",
                "task": task_name,
                "input": row["input"],
                "target": row["target"].strip(),
                "raw": row,
            }
            items.append(item)
    return items


def extract_options(item: dict) -> list[str]:
    """Extract answer options from a BBH item's input text.

    BBH multiple-choice items have options like (A) ... (B) ... (C) ...
    """
    import re
    text = item["input"]
    # Match (A) text, (B) text, etc.
    pattern = r'\(([A-Z])\)\s*([^(]*?)(?=\([A-Z]\)|Options:|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return [(letter, content.strip()) for letter, content in matches]

    # Fallback: just look for option letters in the target to determine format
    # Some BBH tasks have the answer as a full word/phrase
    return []


def build_prompt(item: dict, few_shot: bool = False) -> str:
    """Build the evaluation prompt for a BBH item."""
    # 0-shot by default per spec
    prompt = item["input"].strip()
    if not prompt.endswith("\n"):
        prompt += "\n"
    prompt += "Answer:"
    return prompt


def evaluate_item_logprob(
    model,
    tokenizer,
    item: dict,
    device: str = "cuda",
) -> dict:
    """Evaluate a single BBH item via log-probability comparison.

    Returns dict with selected answer, per-option logprobs, and correctness.
    """
    prompt = build_prompt(item)
    options = extract_options(item)

    if not options:
        # Fallback: treat the target as the only expected answer format
        # For BBH, the target is typically just the option letter like "(A)"
        # Try to extract just the parenthesized options
        target = item["target"].strip()
        # Some tasks use direct answer strings
        # We'll score by comparing generation likelihood of each option
        return _evaluate_direct_answer(model, tokenizer, item, prompt, device)

    option_logprobs = {}
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]

    for letter, content in options:
        # Score the option: compute mean log-prob of option tokens
        # Use the letter as the answer token (most BBH tasks expect just the letter)
        answer_text = f" ({letter})"
        full_text = prompt + answer_text
        full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # Get log-probs for the answer tokens (after the prompt)
        answer_token_ids = full_ids[0, prompt_len:]
        n_answer_tokens = len(answer_token_ids)

        if n_answer_tokens == 0:
            option_logprobs[letter] = float("-inf")
            continue

        # Log-probs at positions [prompt_len-1, ..., seq_len-2] predict tokens at [prompt_len, ..., seq_len-1]
        relevant_logits = logits[0, prompt_len - 1 : prompt_len - 1 + n_answer_tokens]
        log_probs = F.log_softmax(relevant_logits, dim=-1)

        # Gather the log-probs for the actual answer tokens
        token_log_probs = log_probs[range(n_answer_tokens), answer_token_ids].tolist()

        # Arithmetic mean of per-token log-probabilities (spec v0.4)
        mean_logprob = sum(token_log_probs) / n_answer_tokens
        option_logprobs[letter] = mean_logprob

    if not option_logprobs:
        return {
            "selected": None,
            "logprobs": {},
            "correct": False,
            "degenerate": True,
        }

    # Select the option with highest mean log-prob
    selected = max(option_logprobs, key=option_logprobs.get)

    # Determine correctness
    target = item["target"].strip()
    # Target might be "(A)" or just "A" or the full content
    correct = False
    if f"({selected})" in target or selected == target:
        correct = True

    return {
        "selected": selected,
        "logprobs": option_logprobs,
        "correct": correct,
        "degenerate": False,
    }


def _evaluate_direct_answer(model, tokenizer, item, prompt, device):
    """Fallback for items where we can't extract structured options.
    Score each possible answer letter (A-E) directly."""
    import re

    # Try to find what letters are valid from the text
    text = item["input"]
    valid_letters = sorted(set(re.findall(r'\(([A-Z])\)', text)))
    if not valid_letters:
        valid_letters = ["A", "B", "C"]  # reasonable default

    option_logprobs = {}
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]

    for letter in valid_letters:
        answer_text = f" ({letter})"
        full_text = prompt + answer_text
        full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits

        answer_token_ids = full_ids[0, prompt_len:]
        n_answer_tokens = len(answer_token_ids)

        if n_answer_tokens == 0:
            option_logprobs[letter] = float("-inf")
            continue

        relevant_logits = logits[0, prompt_len - 1 : prompt_len - 1 + n_answer_tokens]
        log_probs = F.log_softmax(relevant_logits, dim=-1)
        token_log_probs = log_probs[range(n_answer_tokens), answer_token_ids].tolist()
        mean_logprob = sum(token_log_probs) / n_answer_tokens
        option_logprobs[letter] = mean_logprob

    selected = max(option_logprobs, key=option_logprobs.get) if option_logprobs else None
    target = item["target"].strip()
    correct = selected is not None and (f"({selected})" in target or selected == target)

    return {
        "selected": selected,
        "logprobs": option_logprobs,
        "correct": correct,
        "degenerate": selected is None,
    }


def run_benchmark(
    model,
    tokenizer,
    items: list[dict],
    device: str = "cuda",
    max_items: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Run full benchmark evaluation. Returns aggregate results + per-item details."""
    if max_items:
        items = items[:max_items]

    results = []
    correct_count = 0
    degenerate_count = 0
    start_time = time.time()

    for i, item in enumerate(items):
        item_start = time.time()
        result = evaluate_item_logprob(model, tokenizer, item, device)
        item_time = time.time() - item_start

        result["item_id"] = item["item_id"]
        result["task"] = item["task"]
        result["wall_clock_seconds"] = item_time
        results.append(result)

        if result["correct"]:
            correct_count += 1
        if result.get("degenerate"):
            degenerate_count += 1

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            acc = correct_count / (i + 1) * 100
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(items)}] acc={acc:.1f}% rate={rate:.2f} items/s")

    total_time = time.time() - start_time
    n = len(results)
    accuracy = correct_count / n if n > 0 else 0.0

    return {
        "accuracy": accuracy,
        "accuracy_pct": accuracy * 100,
        "correct_count": correct_count,
        "total_items": n,
        "degenerate_count": degenerate_count,
        "wall_clock_seconds": total_time,
        "seconds_per_item": total_time / n if n > 0 else 0,
        "per_item": results,
    }


def load_model_and_tokenizer(model_id: str, device: str = "cuda", revision: str = None):
    """Load model and tokenizer with fp16 precision."""
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16,
        device_map=device,
        # Note: using default SDPA attention. Eager causes NaN with Qwen2.5 in fp16.
        # Determinism verified empirically in Stage 2.
    )
    model.eval()

    # Report VRAM usage
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
        print(f"  Headroom: {total - reserved:.2f}GB")

    return model, tokenizer
