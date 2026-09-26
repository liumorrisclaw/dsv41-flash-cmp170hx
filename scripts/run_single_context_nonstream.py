#!/usr/bin/env python3
"""Single long-context probe with a non-streaming response for clean EOF."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request

WORDS = "system kernel memory buffer thread process socket packet register cache pointer allocate schedule interrupt virtual physical address translate compile execute branch predict pipeline vector matrix tensor gradient cluster network storage device driver module segment offset boundary".split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(31000 + args.target)
    text = "[single-context-%d] " % args.target + " ".join(rng.choice(WORDS) for _ in range(max(1, args.target - 8)))
    payload = {"model": args.model, "prompt": text, "max_tokens": args.max_tokens, "temperature": 0, "ignore_eos": True}
    key = os.environ.get("STRICT_API_KEY", "")
    req = urllib.request.Request(f"{args.base_url.rstrip('/')}/v1/completions", json.dumps(payload).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            data = json.load(response)
        elapsed = time.perf_counter() - started
        usage = data.get("usage", {})
        row = {"targetContextTokens": args.target, "promptTokens": usage.get("prompt_tokens"), "completionTokens": usage.get("completion_tokens"), "wallSeconds": round(elapsed, 3), "decodeTokensPerSecond": round((usage.get("completion_tokens", 0) - 1) / elapsed, 2) if usage.get("completion_tokens") else None, "validity": "PASS" if usage.get("completion_tokens") == args.max_tokens else "PARTIAL"}
    except Exception as exc:  # noqa: BLE001
        row = {"targetContextTokens": args.target, "validity": "ERROR", "error": f"{type(exc).__name__}: {str(exc)[:240]}"}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False, indent=2)
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
