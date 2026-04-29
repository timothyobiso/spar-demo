"""Plot probe-trajectory pilot output.

Two artifact families:
  1. Per-trace sparklines: probe projection vs token, with sentence
     boundaries and horizon color-coding overlaid. The killer plot.
  2. Aggregate tracking: scatter of per-sentence projection vs labeled
     horizon log-days, plus Spearman ρ for the continuous probe vs the
     null distribution from random-direction controls.

Run:
  python3 experiments/plot_probe_trajectory.py \\
      --in results/probe_traj_pilot.jsonl \\
      --out-dir results/probe_traj_figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HORIZON_COLORS = {
    "tonight":   "#9333ea",  # purple
    "tomorrow":  "#3b82f6",  # blue
    "one_week":  "#06b6d4",  # cyan
    "one_month": "#10b981",  # green
    "one_year":  "#f59e0b",  # amber
    "a_decade":  "#ef4444",  # red
}
HORIZON_ORDER = ["tonight", "tomorrow", "one_week", "one_month", "one_year", "a_decade"]


def load(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------- per-trace sparkline ---------------------------------------

def plot_sparkline(record: dict, out_path: Path):
    proj = np.array(record["projection"])
    ctrl = np.array(record["control_projection"])  # (K, T)
    sentences = record["sentences"]

    T = len(proj)
    fig, (ax_top, ax) = plt.subplots(
        2, 1, figsize=(14, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 4]},
    )

    # Top strip: sentence rectangles colored by horizon
    for s in sentences:
        t_lo, t_hi = s["tok_range"]
        color = HORIZON_COLORS.get(s["horizon_primary"], "#cbd5e1")
        alpha = 0.85 if s["horizon_primary"] else 0.25
        ax_top.barh(0, t_hi - t_lo, left=t_lo, color=color, height=0.8, alpha=alpha)
    ax_top.set_yticks([])
    ax_top.set_xlim(0, T)
    ax_top.set_title(
        f"{record['id']}  (L{record['layer']}, prompt+gen={T} tokens)",
        fontsize=10, loc="left",
    )
    # Legend for horizon colors
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for h, c in HORIZON_COLORS.items()]
    ax_top.legend(handles, list(HORIZON_COLORS.keys()),
                  fontsize=7, loc="upper right", ncol=6, frameon=False)

    # Main panel: per-token projection
    ctrl_mean = ctrl.mean(axis=0)
    ctrl_std = ctrl.std(axis=0)
    ax.fill_between(np.arange(T), ctrl_mean - ctrl_std, ctrl_mean + ctrl_std,
                    color="gray", alpha=0.18, label="random control ±1σ")
    ax.plot(np.arange(T), ctrl_mean, color="gray", linewidth=0.8, alpha=0.5)
    ax.plot(np.arange(T), proj, color="#1F6FEB", linewidth=1.4,
            label="continuous probe")

    # Sentence boundaries (faint vertical lines)
    for s in sentences:
        ax.axvline(s["tok_range"][1], color="black", alpha=0.06, linewidth=0.4)

    # Prompt-end marker
    prompt_end_tok = next(
        (s["tok_range"][0] for s in sentences if not s["in_prompt"]),
        None,
    )
    if prompt_end_tok is not None:
        ax.axvline(prompt_end_tok, color="black", linestyle="--",
                   linewidth=1.0, alpha=0.7, label="prompt → generation")

    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.set_xlim(0, T)
    ax.set_xlabel("token index")
    ax.set_ylabel("residual · probe direction")
    ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ---------------- aggregate tracking ----------------------------------------

def collect_pairs(records: list[dict]) -> list[dict]:
    """Per-sentence rows, generated CoT only (skip prompt sentences)."""
    pairs = []
    for r in records:
        for s in r["sentences"]:
            if s.get("in_prompt"):
                continue
            if s["horizon_log_days"] is None or s["proj_mean"] is None:
                continue
            pairs.append({
                "trace_id": r["id"],
                "kind": r["kind"],
                "horizon": s["horizon_primary"],
                "horizon_log_days": s["horizon_log_days"],
                "proj_mean": s["proj_mean"],
                "ctrl_proj_mean": s.get("ctrl_proj_mean") or [],
                "text": s["text"],
            })
    return pairs


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pure-numpy Spearman (avoid scipy dep). Returns (rho, p≈None)."""
    if len(x) < 3:
        return float("nan"), float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    # crude two-sided p via Fisher z; good enough for a sanity print
    n = len(x)
    if abs(rho) >= 1.0 or n < 4:
        return rho, float("nan")
    z = 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(n - 3)
    from math import erfc
    p = float(erfc(abs(z) / np.sqrt(2)))
    return rho, p


def plot_aggregate(records: list[dict], out_path: Path):
    pairs = collect_pairs(records)
    if not pairs:
        print("[plot] no labeled CoT sentences — nothing to aggregate")
        return None

    proj = np.array([p["proj_mean"] for p in pairs])
    horiz = np.array([p["horizon_log_days"] for p in pairs])
    rho_main, p_main = spearman(proj, horiz)

    K = len(pairs[0]["ctrl_proj_mean"])
    rho_ctrls = []
    for k in range(K):
        ctrl_k = np.array([p["ctrl_proj_mean"][k] for p in pairs])
        rk, _ = spearman(ctrl_k, horiz)
        rho_ctrls.append(rk)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: scatter — projection vs horizon, color-coded
    for h in HORIZON_ORDER:
        xs = [p["proj_mean"] for p in pairs if p["horizon"] == h]
        ys = [p["horizon_log_days"] for p in pairs if p["horizon"] == h]
        if xs:
            # jitter y so dots aren't stacked
            jitter = (np.random.default_rng(hash(h) & 0xFFFF).standard_normal(len(ys)) * 0.06)
            ax1.scatter(xs, np.array(ys) + jitter, color=HORIZON_COLORS[h],
                        label=h, alpha=0.75, s=42, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("mean probe projection per sentence")
    ax1.set_ylabel("labeled horizon (log10 days)")
    ax1.set_title(f"Probe projection vs reasoned horizon (n={len(pairs)} sentences)")
    ax1.legend(fontsize=8, loc="best")
    ax1.axhline(0, color="black", linewidth=0.3, alpha=0.4)

    # Right: rho_main vs control distribution
    ax2.hist(rho_ctrls, bins=max(5, K // 2), color="gray", alpha=0.55,
             edgecolor="white", label=f"random controls (n={K})")
    ax2.axvline(rho_main, color="#1F6FEB", linewidth=2.5,
                label=f"continuous probe  ρ={rho_main:+.2f}")
    ax2.axvline(0, color="black", linewidth=0.3, alpha=0.4)
    ax2.set_xlabel("Spearman ρ  (sentence projection ↔ horizon)")
    ax2.set_ylabel("count")
    ax2.set_title("Tracking signal vs random-direction null")
    ax2.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    print(f"[plot]   continuous: ρ={rho_main:+.3f} (p≈{p_main:.2g})")
    print(f"[plot]   controls:   mean ρ={np.mean(rho_ctrls):+.3f}  "
          f"std={np.std(rho_ctrls):.3f}  max|ρ|={np.max(np.abs(rho_ctrls)):.3f}")
    return {
        "n_pairs": len(pairs),
        "rho_continuous": rho_main,
        "rho_controls": rho_ctrls,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out-dir", default="results/probe_traj_figs")
    args = p.parse_args()

    records = load(Path(args.inp))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot] loaded {len(records)} traces from {args.inp}")

    for r in records:
        slug = r["id"].replace("/", "__")
        plot_sparkline(r, out_dir / f"sparkline_{slug}.png")

    plot_aggregate(records, out_dir / "aggregate_tracking.png")


if __name__ == "__main__":
    main()
