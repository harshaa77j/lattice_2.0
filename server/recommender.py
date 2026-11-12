import json
import logging
import math
import os
import re

import requests

log = logging.getLogger("lattice.recommender")

MODEL_FACTS = {
    "cnn_small": {"cpu_images_per_s": 1085, "cifar_best": 76.1},
    "cnn_bn":    {"cpu_images_per_s": 284,  "cifar_best": 87.9},
    "resnet20":  {"cpu_images_per_s": 195,  "cifar_best": 88.5},
}
NEAR_TIE_POINTS = 1.0      # accuracy differences below this (one run each) are treated as a tie
GPU_SPEEDUP = 3.0          # Apple-GPU / CUDA volunteers vs. CPU, measured roughly on the Mac
ROUND_OVERHEAD_S = 10.0    # transfers, polling and server evaluation per round (measured ~7-10 s)
SMALL_PER_CLASS = 400      # fewer training images per class than this: deeper models may overfit

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def _estimate(model: str, train_count: int, devices: list, rounds: int, local_epochs: int) -> dict:
    """Round time is set by the slowest volunteer training its equal share of the data."""
    n = max(len(devices), 1)
    share = math.ceil(train_count / n)
    speeds = [MODEL_FACTS[model]["cpu_images_per_s"] * (GPU_SPEEDUP if d in ("mps", "cuda") else 1.0)
              for d in (devices or ["cpu"])]
    round_s = share * local_epochs / min(speeds) + ROUND_OVERHEAD_S
    return {"model": model, "est_round_s": round(round_s), "est_total_min": round(round_s * rounds / 60, 1),
                "cifar_best": MODEL_FACTS[model]["cifar_best"]}

def recommend(dataset_meta: dict, devices: list, priority: str = "balanced",
              rounds: int = 20, local_epochs: int = 1, budget_min=None, explain_it: bool = True) -> dict:
    ranking = [_estimate(m, dataset_meta["train_count"], devices, rounds, local_epochs) for m in MODEL_FACTS]
    per_class = dataset_meta["train_count"] / max(dataset_meta["num_classes"], 1)
    small = per_class < SMALL_PER_CLASS
    fits = [r for r in ranking if budget_min is None or r["est_total_min"] <= budget_min] or \
           [min(ranking, key=lambda r: r["est_total_min"])]
    reasons = [f"{dataset_meta['train_count']:,} training images in {dataset_meta['num_classes']} classes "
               f"(about {per_class:.0f} per class)",
               f"{len(devices) or 'no'} volunteer(s) online" + (f": {', '.join(devices)}" if devices else "")]

    if priority == "fastest":
        choice = min(fits, key=lambda r: r["est_total_min"])
        reasons.append("you asked for the fastest result")
    elif priority == "accurate" and not small:
        choice = max(fits, key=lambda r: r["cifar_best"])
        runner_up = max((r for r in fits if r is not choice), key=lambda r: r["cifar_best"], default=None)
        if runner_up and choice["cifar_best"] - runner_up["cifar_best"] < NEAR_TIE_POINTS:
            reasons.append(f"you asked for the most accurate model: on CIFAR-10 it measured {choice['cifar_best']}% "
                           f"vs {runner_up['cifar_best']}% for {runner_up['model']}, almost a tie, "
                           f"but {runner_up['model']} is faster")
        else:
            reasons.append("you asked for the most accurate model, and the dataset is large enough for a deeper network")
    else:
        # Balanced, or a small dataset: the regularised mid-size model, if it fits the budget.
        choice = next((r for r in fits if r["model"] == "cnn_bn"), max(fits, key=lambda r: r["cifar_best"]))
        if small:
            reasons.append("with few images per class, a deeper network is more likely to overfit, "
                       "so a shallower model with augmentation is the safer choice")
        else:
            reasons.append("it balances accuracy against training time")
    if budget_min is not None:
        reasons.append(f"it fits your time budget of {budget_min} minutes" if choice["est_total_min"] <= budget_min
                       else f"no model fits {budget_min} minutes; this is the quickest option")
    return {"model": choice["model"], "ranking": ranking, "reasons": reasons, "small_dataset": small,
                       "explanation": explain(choice, ranking, reasons) if explain_it else None}

def explain(choice: dict, ranking: list, reasons: list) -> str:
    fallback = (f"Recommended: {choice['model']} (about {choice['est_round_s']} s per round, "
                f"~{choice['est_total_min']} min in total), because " + "; ".join(reasons[2:] or reasons) + ".")
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return fallback
    # Plain labels instead of raw ranks: a model can't misread "highest" the way it can misread "3".
    readable = [{"model": r["model"], "seconds_per_round": r["est_round_s"], "total_minutes": r["est_total_min"],
             "measured_cifar10_accuracy_percent": r["cifar_best"]} for r in ranking]
    facts = {"recommended_model": choice["model"], "all_models": readable, "reasons": reasons}
    try:
        r = requests.post(GROQ_URL, timeout=8, headers={"Authorization": f"Bearer {key}"}, json={
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            "temperature": 0.2, "max_tokens": 500,
            "messages": [
                {"role": "system", "content": "You explain a machine-learning model choice to a researcher in 2-3 "
                 "short, friendly sentences. Use ONLY the facts given. Do not invent numbers, models or claims. "
                 "Accuracies were measured on CIFAR-10, not on this dataset: present them as a guide, not a promise. "
                 "'likely'. Times are estimates. No markdown."},
                {"role": "user", "content": json.dumps(facts)},
            ]})
        if r.status_code != 200:
            log.warning("Groq returned %s: %s", r.status_code, r.text[:300])
            return fallback
        text = r.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"\*\*|`", "", text)                     # bold and code marks; underscores stay
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)  # heading marks at the start of a line
        text = re.sub(r"[\u00a0\u202f\u2007]", " ", text)       # typographic spaces -> plain spaces
        if not text:
            log.warning("Groq returned an empty explanation; using the built-in one")
            return fallback
        return text
    except Exception as e:                   # network, timeout, unexpected answer: never fatal
        log.warning("Groq explanation unavailable (%s); using the built-in one", type(e).__name__)
        return fallback