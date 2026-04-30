"""Lexical-dissociation analysis on anchored L22 data.

For each generated sentence in an anchored chained run, ground-truth
label = the prompt's anchor horizon. The sentence either contains a
horizon-word matching its anchor (or any horizon word) — has_lexical=True
— or it doesn't.

Test: within each anchor, do without-word sentences still separate by
horizon as cleanly as with-word sentences? If yes, the L22 probe has a
non-lexical signal. If without-word sentences are flat, the probe is
purely a horizon-word detector.

This is the cheap dissociation test (no GPU). Run:
  python3 experiments/lexical_dissociation.py \\
      --in results/probe_traj_anchored_L22.jsonl \\
      --out-dir results/probe_traj_anchored_L22_figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HORIZON_ORDER = ["tonight", "tomorrow", "one_week", "one_month", "one_year", "a_decade"]
HORIZON_COLORS = {
    "tonight":   "#9333ea", "tomorrow":  "#3b82f6", "one_week":  "#06b6d4",
    "one_month": "#10b981", "one_year":  "#f59e0b", "a_decade":  "#ef4444",
}
HORIZON_LOG_DAYS = {
    "tonight":   -0.602, "tomorrow":   0.000, "one_week":   0.845,
    "one_month":  1.477, "one_year":   2.562, "a_decade":   3.562,
}


def load(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def collect(records: list[dict]):
    """Return one dict per generated sentence with: anchor, has_word, proj, ctrl_proj."""
    rows = []
    for r in records:
        anchor = r.get("anchor_horizon")
        if anchor not in HORIZON_LOG_DAYS:
            continue
        for s in r.get("sentences", []):
            if s.get("in_prompt"):
                continue
            if s["proj_mean"] is None:
                continue
            has_word = bool(s.get("horizon_regex"))
            rows.append({
                "anchor": anchor,
                "has_word": has_word,
                "proj": s["proj_mean"],
                "ctrl_proj": s.get("ctrl_proj_mean") or [],
                "text": s.get("text", "")[:120],
            })
    return rows


def spearman(x, y):
    if len(x) < 3:
        return float("nan"), float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    if abs(rho) >= 1.0 or n < 4:
        return rho, float("nan")
    z = 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(n - 3)
    from math import erfc
    p = float(erfc(abs(z) / np.sqrt(2)))
    return rho, p


def per_anchor_stats(rows):
    out = {}
    for h in HORIZON_ORDER:
        g_with    = [r["proj"] for r in rows if r["anchor"] == h and r["has_word"]]
        g_without = [r["proj"] for r in rows if r["anchor"] == h and not r["has_word"]]
        out[h] = {
            "n_with":    len(g_with),
            "n_without": len(g_without),
            "mean_with":    float(np.mean(g_with))    if g_with    else float("nan"),
            "mean_without": float(np.mean(g_without)) if g_without else float("nan"),
            "sem_with":    float(np.std(g_with)    / max(1, np.sqrt(len(g_with))))    if g_with    else float("nan"),
            "sem_without": float(np.std(g_without) / max(1, np.sqrt(len(g_without)))) if g_without else float("nan"),
        }
    return out


def plot(rows, stats, out_path: Path):
    """Two panels:
       (left) per-anchor with vs without-word means with SEM
       (right) per-condition Spearman ρ vs horizon log-days
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(HORIZON_ORDER))
    width = 0.36
    means_with    = [stats[h]["mean_with"]    for h in HORIZON_ORDER]
    means_without = [stats[h]["mean_without"] for h in HORIZON_ORDER]
    sems_with     = [stats[h]["sem_with"]     for h in HORIZON_ORDER]
    sems_without  = [stats[h]["sem_without"]  for h in HORIZON_ORDER]

    # Bar colors per anchor (same color, two shades)
    for i, h in enumerate(HORIZON_ORDER):
        c = HORIZON_COLORS[h]
        ax1.bar(i - width / 2, means_with[i], width,
                yerr=sems_with[i], color=c, alpha=0.95,
                edgecolor="white", capsize=3,
                label="with horizon word" if i == 0 else None)
        ax1.bar(i + width / 2, means_without[i], width,
                yerr=sems_without[i], color=c, alpha=0.45,
                edgecolor="white", hatch="///", capsize=3,
                label="without horizon word" if i == 0 else None)
        # Annotate counts
        ax1.text(i - width / 2, max(means_with[i], 0) + 0.05,
                 f"n={stats[h]['n_with']}", ha="center", va="bottom", fontsize=7)
        ax1.text(i + width / 2, max(means_without[i], 0) + 0.05,
                 f"n={stats[h]['n_without']}", ha="center", va="bottom", fontsize=7)

    ax1.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(HORIZON_ORDER, fontsize=9, rotation=15, ha="right")
    ax1.set_ylabel("mean probe projection")
    ax1.set_title("Per-anchor probe projection: with vs. without horizon word in sentence")
    ax1.legend(loc="upper left", fontsize=9)

    # Right: per-condition Spearman vs log_days
    proj_with    = np.array([r["proj"] for r in rows if r["has_word"]])
    horiz_with   = np.array([HORIZON_LOG_DAYS[r["anchor"]] for r in rows if r["has_word"]])
    proj_without = np.array([r["proj"] for r in rows if not r["has_word"]])
    horiz_without = np.array([HORIZON_LOG_DAYS[r["anchor"]] for r in rows if not r["has_word"]])

    rho_with, p_with       = spearman(proj_with,    horiz_with)
    rho_without, p_without = spearman(proj_without, horiz_without)

    ax2.scatter(proj_with, horiz_with + np.random.default_rng(0).standard_normal(len(horiz_with)) * 0.04,
                color="#1F6FEB", alpha=0.5, s=20, edgecolor="none",
                label=f"with word (n={len(proj_with)}, ρ={rho_with:+.2f})")
    ax2.scatter(proj_without, horiz_without + np.random.default_rng(1).standard_normal(len(horiz_without)) * 0.04,
                color="#dc2626", alpha=0.5, s=20, edgecolor="none",
                label=f"without word (n={len(proj_without)}, ρ={rho_without:+.2f})")
    ax2.axvline(0, color="black", linewidth=0.3, alpha=0.4)
    ax2.set_xlabel("probe projection (per sentence)")
    ax2.set_ylabel("anchor log_days")
    ax2.set_title("Probe projection vs anchor — does without-word still track horizon?")
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[lex] wrote {out_path}")
    return rho_with, p_with, rho_without, p_without


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out-dir", default="results/probe_traj_anchored_L22_figs")
    args = p.parse_args()

    records = load(Path(args.inp))
    print(f"[lex] loaded {len(records)} traces from {args.inp}")
    rows = collect(records)
    print(f"[lex] {len(rows)} generated sentences across anchors")

    stats = per_anchor_stats(rows)
    print()
    print(f"[lex] {'anchor':>10s}  {'n_with':>7s} {'n_no':>5s}  "
          f"{'mean_with':>10s}  {'mean_no':>10s}  {'Δ(with−no)':>11s}")
    for h in HORIZON_ORDER:
        s = stats[h]
        delta = (s["mean_with"] - s["mean_without"]
                 if not (np.isnan(s["mean_with"]) or np.isnan(s["mean_without"]))
                 else float("nan"))
        print(f"[lex] {h:>10s}  {s['n_with']:>7d} {s['n_without']:>5d}  "
              f"{s['mean_with']:>+10.3f}  {s['mean_without']:>+10.3f}  {delta:>+11.3f}")

    out_dir = Path(args.out_dir)
    rho_with, p_with, rho_without, p_without = plot(
        rows, stats, out_dir / "lexical_dissociation.png"
    )
    print()
    print(f"[lex] Spearman per condition (sentence projection vs anchor log_days):")
    print(f"[lex]   with horizon word:    ρ = {rho_with:+.3f}  (p≈{p_with:.2g})")
    print(f"[lex]   without horizon word: ρ = {rho_without:+.3f}  (p≈{p_without:.2g})")
    print(f"[lex]   ratio: ρ_no/ρ_with = {rho_without / rho_with:.2f}")
    print()
    print("[lex] interpretation guide:")
    print("[lex]   if ρ_without is close to ρ_with → non-lexical signal exists")
    print("[lex]   if ρ_without is much smaller / near zero → probe is mostly lexical")


if __name__ == "__main__":
    main()
