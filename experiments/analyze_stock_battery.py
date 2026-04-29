"""Analyze stock-battery JSONL output: extract numeric predictions, plot.

Run:
  python3 experiments/analyze_stock_battery.py \\
      --in results/stock_battery_v1.jsonl \\
      --out-dir results/stock_battery_figs
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----- price extraction ----------------------------------------------------

# Strict: matches \boxed{<number>} or \boxed{$<number>}.
_BOXED_RE = re.compile(r"\\boxed\{\s*\$?\s*(\d{1,4}(?:,\d{3})*(?:\.\d+)?)\s*\}")
# Fallback: first decimal-or-int (used when \boxed{} is missing).
_PRICE_RE = re.compile(r"\$?\s*(\d{1,4}(?:,\d{3})*(?:\.\d+)?)")
_PERIOD_TOKENS = {
    "tonight":   ["tonight"],
    "tomorrow":  ["tomorrow", "next day"],
    "one_week":  ["next week", "in a week", "week from", "seven days"],
    "one_month": ["next month", "in a month", "month from", "thirty days"],
    "one_year":  ["next year", "in a year", "year from", "twelve months"],
    "a_decade":  ["decade", "ten years", "10 years"],
}


def _try(s: str) -> float | None:
    try:
        v = float(s.replace(",", ""))
        return v if 0.01 <= v <= 100000 else None
    except ValueError:
        return None


def extract_price(text: str) -> tuple[float | None, str]:
    """Try \\boxed{} first, fall back to first sensible number.

    Returns (price, source) where source ∈ {"boxed", "first", "none"}.
    """
    text = text or ""
    m = _BOXED_RE.search(text)
    if m:
        v = _try(m.group(1))
        if v is not None:
            return v, "boxed"
    m = _PRICE_RE.search(text)
    if m:
        v = _try(m.group(1))
        if v is not None:
            return v, "first"
    return None, "none"


def mention_flags(text: str) -> dict[str, bool]:
    t = (text or "").lower()
    return {p: any(tok in t for tok in toks) for p, toks in _PERIOD_TOKENS.items()}


# ----- loading -------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            price, source = extract_price(r["generation"])
            r["price_pred"] = price
            r["price_source"] = source
            cur = r["stock"]["approx_price"]
            r["ratio"] = (r["price_pred"] / cur) if r["price_pred"] else None
            r["mentions"] = mention_flags(r["generation"])
            out.append(r)
    return out


# ----- plots ---------------------------------------------------------------

def plot_steering_by_horizon(records, out_dir: Path):
    """Bar chart per horizon: mean predicted-ratio for each steering config."""
    horizons = sorted({r["horizon"]["days"] for r in records})
    steerings = sorted({r["steering"]["name"] for r in records})

    fig, axes = plt.subplots(1, len(horizons), figsize=(3.6 * len(horizons), 5),
                             sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        means, errs, labels = [], [], []
        for s in steerings:
            ratios = [r["ratio"] for r in records
                      if r["horizon"]["days"] == h
                      and r["steering"]["name"] == s
                      and r["ratio"] is not None]
            if not ratios:
                continue
            means.append(float(np.mean(ratios)))
            errs.append(float(np.std(ratios) / max(1, np.sqrt(len(ratios)))))
            labels.append(s)
        if not means:
            continue
        y = np.arange(len(labels))
        ax.barh(y, means, xerr=errs, color="#1F6FEB", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(1.0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"horizon: {h} days", fontsize=10)
        ax.set_xlabel("predicted / current")
    fig.suptitle("Predicted-price ratio by steering config × stated horizon")
    plt.tight_layout()
    plt.savefig(out_dir / "steering_by_horizon.png", dpi=150)
    plt.close(fig)
    print(f"[analyze] wrote {out_dir/'steering_by_horizon.png'}")


def plot_strength_curve(records, out_dir: Path):
    """Continuous-mode and interp-mode dose-response, if available."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, mode in zip(axes, ("interp", "continuous")):
        # Pick records where steering is mode/t=X.X/s=NNN
        prefix = f"{mode}/"
        rels = [r for r in records if r["steering"]["name"].startswith(prefix)
                and r["ratio"] is not None]
        if not rels:
            ax.set_title(f"{mode}: no data")
            continue
        # Group by (time, strength)
        cells: dict[tuple[float, float], list[float]] = defaultdict(list)
        for r in rels:
            t = float(r["steering"].get("time", 0.0))
            s = float(r["steering"].get("strength", 0.0))
            cells[(t, s)].append(r["ratio"])
        # Plot one line per time value
        times = sorted({k[0] for k in cells})
        for t in times:
            xs = sorted({k[1] for k in cells if k[0] == t})
            ys = [float(np.mean(cells[(t, s)])) for s in xs]
            ax.plot(xs, ys, marker="o", label=f"t={t}")
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"{mode} mode")
        ax.set_xlabel("strength α")
        ax.set_ylabel("predicted / current")
        ax.legend(fontsize=8)
    fig.suptitle("Dose-response: predicted-price ratio vs steering strength")
    plt.tight_layout()
    plt.savefig(out_dir / "strength_curve.png", dpi=150)
    plt.close(fig)
    print(f"[analyze] wrote {out_dir/'strength_curve.png'}")


def plot_real_vs_fake(records, out_dir: Path):
    """Mean predicted-ratio split by real vs fake, by steering config."""
    steerings = sorted({r["steering"]["name"] for r in records})
    pairs = []
    for s in steerings:
        real_ratios = [r["ratio"] for r in records
                       if r["steering"]["name"] == s
                       and r["stock"]["real"]
                       and r["ratio"] is not None]
        fake_ratios = [r["ratio"] for r in records
                       if r["steering"]["name"] == s
                       and not r["stock"]["real"]
                       and r["ratio"] is not None]
        if not (real_ratios and fake_ratios):
            continue
        pairs.append((s, np.mean(real_ratios), np.mean(fake_ratios)))

    if not pairs:
        return
    pairs.sort(key=lambda x: x[1] - x[2], reverse=True)
    labels = [p[0] for p in pairs]
    real_means = [p[1] for p in pairs]
    fake_means = [p[2] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(labels) + 1.2))
    y = np.arange(len(labels))
    ax.barh(y - 0.18, real_means, height=0.36, label="real",  color="#1F6FEB")
    ax.barh(y + 0.18, fake_means, height=0.36, label="fake",  color="#15803D")
    ax.axvline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted / current")
    ax.legend()
    ax.set_title("Real vs fake stocks: mean predicted-ratio per steering config")
    plt.tight_layout()
    plt.savefig(out_dir / "real_vs_fake.png", dpi=150)
    plt.close(fig)
    print(f"[analyze] wrote {out_dir/'real_vs_fake.png'}")


def plot_background_effect(records, out_dir: Path):
    """For baseline + a few key steering configs, compare across backgrounds."""
    bgs = sorted({r["background"] for r in records})
    if len(bgs) < 2:
        return
    target_steerings = ["baseline", "periods/tonight/+60", "periods/a_decade/+60",
                        "interp/t=5.0/s=60", "continuous/t=5.0/s=60"]
    target_steerings = [s for s in target_steerings
                        if any(r["steering"]["name"] == s for r in records)]
    if not target_steerings:
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(target_steerings)), 4.5))
    width = 0.8 / len(bgs)
    x = np.arange(len(target_steerings))
    for i, bg in enumerate(bgs):
        means = []
        for s in target_steerings:
            ratios = [r["ratio"] for r in records
                      if r["steering"]["name"] == s
                      and r["background"] == bg
                      and r["ratio"] is not None]
            means.append(float(np.mean(ratios)) if ratios else float("nan"))
        ax.bar(x + (i - (len(bgs) - 1) / 2) * width, means, width=width, label=bg)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(target_steerings, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("predicted / current")
    ax.set_title("Background variant × steering config")
    ax.legend(title="background")
    plt.tight_layout()
    plt.savefig(out_dir / "background_effect.png", dpi=150)
    plt.close(fig)
    print(f"[analyze] wrote {out_dir/'background_effect.png'}")


def write_summary_csv(records, out_dir: Path):
    fields = [
        "i", "ticker", "real", "sector", "current_price",
        "horizon_days", "horizon_label",
        "background",
        "steering_name", "steering_mode",
        "seed",
        "predicted_price", "ratio",
        "mentions_period",
        "generation",
    ]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            mp = [k for k, v in r["mentions"].items() if v]
            w.writerow({
                "i": r["i"],
                "ticker": r["stock"]["ticker"],
                "real": r["stock"]["real"],
                "sector": r["stock"]["sector"],
                "current_price": r["stock"]["approx_price"],
                "horizon_days": r["horizon"]["days"],
                "horizon_label": r["horizon"]["label"],
                "background": r["background"],
                "steering_name": r["steering"]["name"],
                "steering_mode": r["steering"]["mode"],
                "seed": r["seed"],
                "predicted_price": r["price_pred"],
                "ratio": r["ratio"],
                "mentions_period": ",".join(mp),
                "generation": r["generation"][:200],
            })
    print(f"[analyze] wrote {out_dir/'summary.csv'} ({len(records)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out-dir", default="results/stock_battery_figs")
    args = p.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] loading {inp}")
    records = load_records(inp)
    n_extracted = sum(1 for r in records if r["price_pred"] is not None)
    n_boxed = sum(1 for r in records if r.get("price_source") == "boxed")
    n_first = sum(1 for r in records if r.get("price_source") == "first")
    print(f"[analyze] {len(records)} records, {n_extracted} with extractable price "
          f"({100*n_extracted/max(1,len(records)):.1f}%)")
    print(f"[analyze]   sourced from boxed: {n_boxed}  · fallback first-number: {n_first}")

    write_summary_csv(records, out_dir)
    plot_steering_by_horizon(records, out_dir)
    plot_real_vs_fake(records, out_dir)
    plot_background_effect(records, out_dir)
    plot_strength_curve(records, out_dir)


if __name__ == "__main__":
    main()
