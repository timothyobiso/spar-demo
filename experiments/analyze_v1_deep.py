"""Deeper analysis on the v1 stock battery JSONL.

Focuses on cross-cutting findings the basic analyzer didn't surface:
  1. Stated × steering heatmap (does the effect compound at long horizons?)
  2. Per-stock × steering heatmap for decade-steered (which tickers are most
     affected by the model's growth priors?)
  3. Period-mention frequency (does steering shift FRAMING even when the
     extracted price doesn't move?)
  4. Real vs fake × horizon interaction (where does the gap appear?)

Run:
  python3 experiments/analyze_v1_deep.py \\
    --in results/stock_battery_v1.jsonl \\
    --out-dir results/stock_battery_figs/deep
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


_PRICE_RE = re.compile(r"\$?\s*(\d{1,4}(?:,\d{3})*(?:\.\d+)?)")
_PERIOD_TOKENS = {
    "tonight":   ["tonight"],
    "tomorrow":  ["tomorrow", "next day"],
    "one_week":  ["next week", "in a week", "week from", "seven days"],
    "one_month": ["next month", "in a month", "month from", "thirty days"],
    "one_year":  ["next year", "in a year", "year from", "twelve months", " 2025", " 2026"],
    "a_decade":  ["decade", "ten years", "10 years"],
}


def extract_price(text: str) -> float | None:
    m = _PRICE_RE.search(text or "")
    if not m:
        return None
    s = m.group(1).replace(",", "")
    try:
        v = float(s)
        return v if 0.01 <= v <= 100000 else None
    except ValueError:
        return None


def sanitize_ratio(ratio: float | None) -> float | None:
    """Drop wild outliers likely caused by the regex picking up year tokens etc."""
    if ratio is None:
        return None
    if ratio < 0.05 or ratio > 50:
        return None
    return ratio


def mention_flags(text: str) -> dict[str, bool]:
    t = (text or "").lower()
    return {p: any(tok in t for tok in toks) for p, toks in _PERIOD_TOKENS.items()}


def load(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            r["price_pred"] = extract_price(r["generation"])
            cur = r["stock"]["approx_price"]
            raw_ratio = (r["price_pred"] / cur) if r["price_pred"] else None
            r["ratio_raw"] = raw_ratio
            r["ratio"] = sanitize_ratio(raw_ratio)
            r["mentions"] = mention_flags(r["generation"])
            out.append(r)
    return out


# ---------------- plots -----------------------------------------------------

def plot_stated_x_steering_heatmap(records, out_dir: Path):
    horizons = sorted({r["horizon"]["days"] for r in records})
    steerings = sorted({r["steering"]["name"] for r in records})

    M = np.full((len(steerings), len(horizons)), np.nan)
    for i, s in enumerate(steerings):
        for j, h in enumerate(horizons):
            cell = [r["ratio"] for r in records
                    if r["steering"]["name"] == s and r["horizon"]["days"] == h
                    and r["ratio"] is not None]
            if cell:
                M[i, j] = float(np.median(cell))

    fig, ax = plt.subplots(figsize=(7, 0.36 * len(steerings) + 1.6))
    norm = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.5)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_yticks(range(len(steerings)))
    ax.set_yticklabels(steerings, fontsize=8)
    ax.set_xlabel("stated horizon")
    ax.set_title("Median predicted-ratio: stated horizon × steering config\n(blue = depressed, red = inflated, white = no change)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v - 1.0) > 0.7 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="predicted/current")
    plt.tight_layout()
    plt.savefig(out_dir / "stated_x_steering_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"[deep] wrote {out_dir/'stated_x_steering_heatmap.png'}")


def plot_per_stock_decade(records, out_dir: Path):
    """For periods/a_decade/+60, show predicted ratio per stock × stated horizon."""
    target = "periods/a_decade/+60"
    rels = [r for r in records if r["steering"]["name"] == target and r["ratio"] is not None]
    if not rels:
        return
    stocks = sorted({r["stock"]["ticker"] for r in rels},
                    key=lambda t: (not next((r["stock"]["real"] for r in rels
                                              if r["stock"]["ticker"] == t), False), t))
    horizons = sorted({r["horizon"]["days"] for r in rels})

    M = np.full((len(stocks), len(horizons)), np.nan)
    for i, t in enumerate(stocks):
        for j, h in enumerate(horizons):
            cell = [r["ratio"] for r in rels
                    if r["stock"]["ticker"] == t and r["horizon"]["days"] == h]
            if cell:
                M[i, j] = float(np.median(cell))

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(stocks) + 1.6))
    norm = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=8.0)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_yticks(range(len(stocks)))
    yt = []
    for s in stocks:
        real = next(r["stock"]["real"] for r in rels if r["stock"]["ticker"] == s)
        yt.append(f"{s} {'(real)' if real else '(fake)'}")
    ax.set_yticklabels(yt, fontsize=9)
    ax.set_xlabel("stated horizon")
    ax.set_title(f"Per-stock predicted-ratio under {target}")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v - 1.0) > 1.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="predicted/current")
    plt.tight_layout()
    plt.savefig(out_dir / "per_stock_decade.png", dpi=150)
    plt.close(fig)
    print(f"[deep] wrote {out_dir/'per_stock_decade.png'}")


def plot_period_mentions(records, out_dir: Path):
    """For each steering config, fraction of generations mentioning each period word."""
    steerings = sorted({r["steering"]["name"] for r in records})
    periods = list(_PERIOD_TOKENS.keys())

    M = np.zeros((len(steerings), len(periods)))
    for i, s in enumerate(steerings):
        cells = [r for r in records if r["steering"]["name"] == s]
        if not cells:
            continue
        for j, p in enumerate(periods):
            M[i, j] = sum(1 for r in cells if r["mentions"][p]) / len(cells)

    fig, ax = plt.subplots(figsize=(7, 0.36 * len(steerings) + 1.6))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=20, ha="right")
    ax.set_yticks(range(len(steerings)))
    ax.set_yticklabels(steerings, fontsize=8)
    ax.set_xlabel("period word in generation")
    ax.set_title("Fraction of generations mentioning each period word\n(reveals framing shifts even when price doesn't change)")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="mention rate")
    plt.tight_layout()
    plt.savefig(out_dir / "period_mention_frequency.png", dpi=150)
    plt.close(fig)
    print(f"[deep] wrote {out_dir/'period_mention_frequency.png'}")


def plot_real_vs_fake_by_horizon(records, out_dir: Path):
    """For three key steerings, show real vs fake ratio at each horizon."""
    targets = ["baseline", "periods/a_decade/+60", "interp/t=5.0/s=60", "continuous/t=5.0/s=60"]
    targets = [t for t in targets if any(r["steering"]["name"] == t for r in records)]
    horizons = sorted({r["horizon"]["days"] for r in records})

    fig, axes = plt.subplots(1, len(targets), figsize=(3.5 * len(targets), 4.5), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, t in zip(axes, targets):
        real_means, fake_means, real_err, fake_err = [], [], [], []
        for h in horizons:
            real = [r["ratio"] for r in records
                    if r["steering"]["name"] == t and r["horizon"]["days"] == h
                    and r["stock"]["real"] and r["ratio"] is not None]
            fake = [r["ratio"] for r in records
                    if r["steering"]["name"] == t and r["horizon"]["days"] == h
                    and not r["stock"]["real"] and r["ratio"] is not None]
            real_means.append(float(np.median(real)) if real else np.nan)
            fake_means.append(float(np.median(fake)) if fake else np.nan)
            real_err.append(float(np.std(real)/max(1,np.sqrt(len(real)))) if real else 0)
            fake_err.append(float(np.std(fake)/max(1,np.sqrt(len(fake)))) if fake else 0)
        x = np.arange(len(horizons))
        ax.errorbar(x - 0.08, real_means, yerr=real_err, marker="o", color="#1F6FEB", label="real", linewidth=2)
        ax.errorbar(x + 0.08, fake_means, yerr=fake_err, marker="s", color="#15803D", label="fake", linewidth=2)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}d" for h in horizons], fontsize=8)
        ax.set_xlabel("stated horizon")
        ax.set_title(t, fontsize=9)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("predicted / current (log)")
    fig.suptitle("Real vs fake stocks across stated horizons, by steering config")
    plt.tight_layout()
    plt.savefig(out_dir / "real_fake_x_horizon.png", dpi=150)
    plt.close(fig)
    print(f"[deep] wrote {out_dir/'real_fake_x_horizon.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out-dir", default="results/stock_battery_figs/deep")
    args = p.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[deep] loading {inp}")
    records = load(inp)
    n_extracted = sum(1 for r in records if r["ratio_raw"] is not None)
    n_clean = sum(1 for r in records if r["ratio"] is not None)
    print(f"[deep] {len(records)} records, {n_extracted} extracted, "
          f"{n_clean} after sanitizing (dropped {n_extracted - n_clean} outliers)")

    plot_stated_x_steering_heatmap(records, out_dir)
    plot_per_stock_decade(records, out_dir)
    plot_period_mentions(records, out_dir)
    plot_real_vs_fake_by_horizon(records, out_dir)


if __name__ == "__main__":
    main()
