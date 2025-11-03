#!/usr/bin/env python3
"""
CIFAR-10 classification using ai.sooners.us (OpenAI-compatible Chat Completions).

This script:
  • Samples 100 images (10/class) from CIFAR-10 (train split)
  • Sends each image as base64 Data URL to /api/chat/completions (gemma3:4b)
  • Tries multiple system prompts (prompt engineering)
  • Records predictions, computes accuracy, and saves confusion matrices
  • Saves misclassifications for manual inspection

Requirements:
  pip install requests python-dotenv torch torchvision pillow scikit-learn matplotlib
"""

import os
import io
import base64
import random
import json
import csv
import time
import re
from typing import List, Dict, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image
import torch
from torchvision.datasets import CIFAR10
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# ── Load secrets ──────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.expanduser("~"), ".soonerai.env"))
API_KEY = os.getenv("SOONERAI_API_KEY")
BASE_URL = os.getenv("SOONERAI_BASE_URL", "https://ai.sooners.us").rstrip("/")
MODEL = os.getenv("SOONERAI_MODEL", "gemma3:4b")

if not API_KEY:
    raise RuntimeError("Missing SOONERAI_API_KEY in ~/.soonerai.env")

# ── Config ───────────────────────────────────────────────────────────────────
SEED = 1337
SAMPLES_PER_CLASS = 10
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ── SYSTEM PROMPTS TO TEST ───────────────────────────────────────────────────
# Provide at least two different prompts to compare.
PROMPTS = [
    {
        "name": "concise_label_only",
        "system": """
You are a precise image classifier for CIFAR-10. When given an image, reply with exactly one word
that is one of the valid class labels. Do NOT add any other text, explanation, punctuation, or formatting.
Valid labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
If unsure, pick the most likely label from that list.
""".strip()
    },
    {
        "name": "explain_then_label",
        "system": """
You are an image analyst. Briefly (one sentence) note the dominant visual cues (e.g., 'looks like a small bird with wings'), then on a new line output EXACTLY one label chosen from:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
The final output must be a single line containing exactly one of the labels above.
""".strip()
    },
    # You can add more prompts here to experiment.
]

USER_INSTRUCTION = f"""
Classify this CIFAR-10 image. Respond with exactly one label from this list:
{', '.join(CLASSES)}
Your reply must be just the label, nothing else (unless the system prompt explicitly allows one explanatory line).
""".strip()

# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_name(s: str) -> str:
    """Make a filesystem-safe short name for a prompt."""
    s = s.lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_-]", "", s)
    return s[:40]

def pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    """Encode a PIL image to base64 JPEG data URL."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def post_chat_completion_image(
    image_data_url: str,
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    timeout: int = 60,
) -> str:
    """
    Send an image + instruction to /api/chat/completions and return the text reply.
    """
    url = f"{base_url}/api/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")

    data = resp.json()
    # Expecting structure similar to OpenAI chat responses:
    # data["choices"][0]["message"]["content"]
    return data["choices"][0]["message"]["content"].strip()

def normalize_label(text: str) -> str:
    """Map model reply to a valid CIFAR-10 class if possible (simple heuristic)."""
    if not text:
        return "__unknown__"
    t = text.lower().strip()
    # if the model included an explanation and then the label on a new line, pick the last line
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        candidate = lines[-1]
    else:
        candidate = t

    # remove periods/commas/punctuation
    candidate = re.sub(r"[^\w\s]", "", candidate).strip()

    # exact match first
    if candidate in CLASSES:
        return candidate
    # loose matching: pick first class name contained in candidate
    for c in CLASSES:
        if c in candidate:
            return c
    # if none matched, try to find any class token in entire text
    for c in CLASSES:
        if c in t:
            return c
    return "__unknown__"

# ── Data: stratified sample of 100 images (10/class) ─────────────────────────
def stratified_sample_cifar10(root: str = "./data") -> List[Tuple[Image.Image, int]]:
    """
    Download CIFAR-10 (train split) and return a list of (PIL_image, target) pairs:
    exactly SAMPLES_PER_CLASS per class.
    """
    ds = CIFAR10(root=root, train=True, download=True)
    # Build indices per class
    per_class: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(ds):
        per_class[label].append(idx)

    # Sample with fixed seed
    random.seed(SEED)
    selected = []
    for label in range(10):
        chosen = random.sample(per_class[label], SAMPLES_PER_CLASS)
        for idx in chosen:
            img, tgt = ds[idx]
            selected.append((img, tgt))
    return selected

# ── Core evaluation for one prompt ───────────────────────────────────────────
def evaluate_prompt(prompt_name: str, system_prompt: str, samples: List[Tuple[Image.Image, int]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    y_true = []
    y_pred = []
    bad = []

    print(f"\n=== Evaluating prompt: {prompt_name} (saving to {out_dir}) ===")
    for i, (img, tgt) in enumerate(samples, start=1):
        data_url = pil_to_base64_jpeg(img)
        try:
            reply = post_chat_completion_image(
                image_data_url=data_url,
                system_prompt=system_prompt,
                model=MODEL,
                base_url=BASE_URL,
                api_key=API_KEY,
                temperature=0.0,
            )
        except Exception as e:
            print(f"[{i:03d}/100] API error: {e}")
            reply = ""
            pred_label = "__error__"
            pred_idx = -1
        else:
            pred_label = normalize_label(reply)
            pred_idx = CLASSES.index(pred_label) if pred_label in CLASSES else -1

        y_true.append(tgt)
        y_pred.append(pred_idx)

        true_label = CLASSES[tgt]
        print(f"[{i:03d}/100] true={true_label:>10s} | pred={pred_label:>10s} | raw='{reply}'")

        if pred_idx == -1:
            bad.append({
                "i": i,
                "true": true_label,
                "raw_reply": reply,
                "pred_label": pred_label,
            })

    # Build confusion matrix manually so unknowns are accounted as a separate column if desired.
    # Here we coerce unknown predictions into an "unknown" column index = 10 to keep 11 columns
    n_classes = len(CLASSES)
    cm = np.zeros((n_classes, n_classes), dtype=int)  # 10x10 final (unknowns will be counted to last column)
    for t, p in zip(y_true, y_pred):
        if 0 <= p < n_classes:
            cm[t, p] += 1
        else:
            # treat unknowns as predicted as index (n_classes-1) so matrix remains 10x10.
            # This is consistent with the original template behavior (unknown counted as some class).
            cm[t, n_classes - 1] += 1

    # Accuracy: count exact matches only (unknowns are wrong)
    y_pred_for_acc = [p if 0 <= p < n_classes else -1 for p in y_pred]
    correct = sum(1 for t, p in zip(y_true, y_pred_for_acc) if p == t)
    acc = correct / len(y_true)
    print(f"\nPrompt '{prompt_name}' Accuracy over {len(y_true)} images: {acc*100:.2f}%")

    # Save confusion matrix image
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix ({prompt_name})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(n_classes), CLASSES, rotation=45, ha="right")
    plt.yticks(range(n_classes), CLASSES)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            plt.text(c, r, str(cm[r, c]), ha="center", va="center")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=180)
    plt.close()
    print(f"Saved {cm_path}")

    # Save misclassifications
    mis_path = os.path.join(out_dir, "misclassifications.jsonl")
    with open(mis_path, "w") as f:
        for row in bad:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(bad)} misclassifications to {mis_path}")

    # Save a small JSON summary
    summary = {
        "prompt_name": prompt_name,
        "accuracy": acc,
        "n_images": len(y_true),
        "n_mis": len(bad),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Return summary info for aggregated CSV
    return summary

def main():
    samples = stratified_sample_cifar10()
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    summary_rows = []
    for p in PROMPTS:
        name = p.get("name") or safe_name(p["system"][:60])
        name = safe_name(name)
        out_dir = os.path.join(results_dir, name)
        # Run evaluation
        summary = evaluate_prompt(name, p["system"], samples, out_dir)
        summary_rows.append(summary)
        # small pause to avoid accidental rate limits
        time.sleep(0.5)

    # Save overall summary CSV
    csv_path = os.path.join(results_dir, "summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["prompt_name", "accuracy", "n_images", "n_mis"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"\nSaved aggregated summary to {csv_path}")

if __name__ == "__main__":
    main()
