#!/usr/bin/env python3
import os
import sys
from typing import List, Tuple

REQUIRED_FILES = [
    "notebooks/01_finetune_sentiment_llama3_colab.ipynb",
    "scripts/train_sentiment_llama3.py",
    "README.md",
]

RECOMMENDED_KEYWORDS = {
    "scripts/train_sentiment_llama3.py": [
        "BitsAndBytesConfig",
        "LoraConfig",
        "SFTTrainer",
        "amazon_us_reviews",
    ]
}


def check_exists(path: str) -> Tuple[bool, str]:
    ok = os.path.exists(path)
    return ok, f"{'FOUND' if ok else 'MISSING'}: {path}"


def file_contains(path: str, keywords: List[str]) -> Tuple[bool, str]:
    if not os.path.exists(path):
        return False, f"Cannot check keywords; file missing: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        missing = [k for k in keywords if k not in text]
        if missing:
            return False, f"Missing keywords in {path}: {missing}"
        return True, f"All recommended keywords present in {path}"
    except Exception as e:
        return False, f"Error reading {path}: {e}"


def main() -> int:
    print("[PM] Project Manager Agent — Quick Readiness Check")
    all_ok = True

    # Required files
    for p in REQUIRED_FILES:
        ok, msg = check_exists(p)
        all_ok = all_ok and ok
        print(f"[PM] {msg}")

    # Keyword checks
    for p, kws in RECOMMENDED_KEYWORDS.items():
        ok, msg = file_contains(p, kws)
        all_ok = all_ok and ok
        print(f"[PM] {msg}")

    # requirements.txt suggestion
    req_path = "requirements.txt"
    if os.path.exists(req_path):
        print(f"[PM] FOUND: {req_path}")
    else:
        print(f"[PM] WARN: {req_path} not found (recommended for local runs)")

    if all_ok:
        print("[PM] Status: READY for Colab fine-tuning.")
        return 0
    else:
        print("[PM] Status: Issues detected — please address warnings above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


