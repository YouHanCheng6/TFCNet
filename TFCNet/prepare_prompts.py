"""Prepare dataset summaries offline. This module is never called by training.

This is a new public implementation, not a record of the manuscript experiments.
Only rows explicitly identified as training data are read for statistical analysis.
"""
import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def word_count(text):
    """Count whitespace-delimited English words; tokenizer tokens are separate."""
    return len(text.split())


def feature_statistics(values):
    """Descriptive statistics, linear trend and detrended spectral estimates."""
    n = len(values)
    centered = values - values.mean()
    t = np.arange(n, dtype=float)
    slope = float(np.polyfit(t, values, 1)[0])
    residual = centered - slope * (t - t.mean())
    power = np.abs(np.fft.rfft(residual)) ** 2
    # DC is excluded. Require at least two cycles in the training interval.
    power[:2] = 0
    total = float(power.sum())
    k = int(np.argmax(power)) if total > 0 else 0
    constant = bool(np.std(values) < 1e-12)
    if constant:
        k, total = 0, 0.0
    return {
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()), "std": float(values.std()),
        "linear_trend_slope_per_step": slope,
        "dominant_period_steps": float(n / k) if k else None,
        "spectral_peak_fraction": float(power[k] / total) if k else 0.0,
        "constant": constant,
    }


def collect_statistics(args):
    if args.train_rows < 4:
        raise ValueError("train_rows must identify at least four training observations")
    # Avoid even loading the validation/test rows into the statistics pipeline.
    frame = pd.read_csv(args.csv, nrows=args.train_rows)
    if len(frame) != args.train_rows:
        raise ValueError("CSV contains fewer rows than the specified training split")
    if args.columns:
        columns = args.columns
    else:
        columns = [c for c in frame.columns if c != args.date_column]
    if not columns:
        raise ValueError("No feature columns selected")
    data = frame[columns].apply(pd.to_numeric, errors="raise")
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training data contain missing/non-finite values; preprocess explicitly")
    digest = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    source_mean, source_std = values.mean(0), values.std(0)
    if args.standardize:
        values = (values - source_mean) / np.where(source_std > 0, source_std, 1)
    stats = {
        "dataset": args.dataset, "domain": args.domain,
        "sampling_interval": args.sampling_interval,
        "training_rows": args.train_rows, "num_variables": len(columns),
        "value_space": "training-standardized" if args.standardize else "original",
        "training_values_sha256": digest,
        "features": {str(c): feature_statistics(values[:, k])
                     for k, c in enumerate(columns)},
        "estimator_notes": (
            "Trend is an ordinary least-squares slope. Dominant period is the "
            "largest detrended FFT peak with at least two observed cycles. "
            "Peak fraction is a descriptive concentration measure, not a "
            "significance test or proof of seasonality."
        ),
    }
    if args.standardize:
        stats["scaler"] = {"mean": source_mean.tolist(), "std": source_std.tolist()}
    return stats


def request_summary(args, instruction):
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("Set DASHSCOPE_API_KEY before generating summaries")
    body = {"model": args.model, "temperature": 0,
            "max_tokens": max(512, max(args.words) * 4),
            "messages": [{"role": "user", "content": instruction}]}
    request = Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    text = result["choices"][0]["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Qwen returned no summary")
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-rows", type=int, required=True,
                        help="Number of initial rows in the actual training split (excluding header)")
    parser.add_argument("--sampling-interval", required=True, help="e.g. 1 hour, 15 minutes")
    parser.add_argument("--domain", default="", help="Optional factual domain description")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--columns", nargs="+", help="Selected variables; default: all except date")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--words", nargs="+", type=int, default=[20, 40, 80])
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Explicitly overwrite matching cache files")
    args = parser.parse_args()
    if args.attempts < 1 or any(w < 1 for w in args.words):
        parser.error("attempts and word lengths must be positive")
    stats = collect_statistics(args)
    args.output.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", args.dataset)
    for words in dict.fromkeys(args.words):
        instruction = (
            "Analyze the following training-data statistics and produce exactly "
            f"{words} whitespace-separated English words as one dataset-level summary. "
            "Return only the summary, without headings. Describe supported numerical "
            "characteristics, trend and periodic structure relevant to forecasting. "
            "Do not invent domain behavior or treat a spectral peak as proof of seasonality. "
            "Prioritize useful information within the requested length.\n"
            + json.dumps(stats, ensure_ascii=False)
        )
        fingerprint = hashlib.sha256((args.model + instruction).encode()).hexdigest()
        destination = args.output / f"{safe_name}_{words}w.json"
        if destination.exists() and not args.force:
            cached = json.loads(destination.read_text(encoding="utf-8"))
            if (cached.get("fingerprint") != fingerprint
                    or word_count(cached.get("summary", "")) != words):
                raise ValueError(f"Cache differs from requested preparation: {destination}; use a new output directory")
            print(f"Using cache: {destination}")
            continue
        for attempt in range(args.attempts):
            # Each length is generated separately, never by truncating another summary.
            retry = "" if not attempt else "\nEnsure the exact requested English word count."
            summary = request_summary(args, instruction + retry)
            if word_count(summary) == words:
                break
        else:
            raise ValueError(f"Qwen did not meet {words}-word requirement; no cache written for this length")
        record = {"schema_version": 1, "dataset": args.dataset,
                  "generator": args.model, "target_words": words,
                  "actual_words": word_count(summary), "summary": summary,
                  "statistics": stats, "instruction": instruction + retry,
                  "fingerprint": fingerprint,
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(destination)
        print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
