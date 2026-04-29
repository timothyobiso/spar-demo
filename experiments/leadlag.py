"""Lead/lag analysis: at each phase transition in forward chained CoT,
where does the probe shift — *before* the header word arrives (anticipation),
*at* the header (lexical echo), or only in the body content (downstream)?

Time-locks per-token probe projections to each transition (header start),
extracts a window of ±W tokens, then averages across traces and per
transition-type.

Run:
  python3 experiments/leadlag.py \\
      --in results/probe_traj_chained_k30.jsonl \\
      --out results/probe_traj_chained_k30_figs/leadlag.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHASE_ORDER = ["tonight", "tomorrow", "one_week", "one_month", "one_year", "a_decade"]
PHASE_COLORS = {
    "tonight":   "#9333ea", "tomorrow":  "#3b82f6", "one_week":  "#06b6d4",
    "one_month": "#10b981", "one_year":  "#f59e0b", "a_decade":  "#ef4444",
}


def load(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def char_to_token(offsets: list[list[int]], char_pos: int) -> int:
    """Return the token index whose span starts at or just after char_pos."""
    for i, (cs, ce) in enumerate(offsets):
        if cs == 0 and ce == 0:
            continue
        if ce >= char_pos:
            return i
    return len(offsets) - 1


def collect_windows(records: list[dict], window: int = 12) -> dict:
    """For each forward trace, extract probe values in ±window tokens around
    each header-start, plus the to-phase label and a control-direction
    average for the same window.

    Returns dict[from_label -> dict[to_label -> list of arrays of shape (2W+1,)]]
    plus a parallel control structure.
    """
    transitions = {}  # (from_label, to_label) -> list of (probe_window, ctrl_window_avg)
    for r in records:
        if r.get("reverse") or r.get("random_order"):
            continue
        proj = np.array(r["projection"])
        ctrl = np.array(r["control_projection"])  # (K, T)
        offsets = r["token_offsets"]
        phases = r["phases"]
        T = len(proj)
        for i in range(1, len(phases)):
            from_lbl = phases[i - 1]["label"]
            to_lbl = phases[i]["label"]
            # Recompute header start: body_start - len("\n\n" + header + "\n")
            ph = phases[i]
            if "char_header_start" in ph:
                header_char = ph["char_header_start"]
            else:
                header_char = ph["char_body_start"] - (len(ph["header"]) + 3)
            tok_at_header = char_to_token(offsets, header_char)
            lo = tok_at_header - window
            hi = tok_at_header + window + 1
            if lo < 0 or hi > T:
                continue
            probe_win = proj[lo:hi].copy()
            ctrl_win  = ctrl[:, lo:hi].mean(axis=0)  # (2W+1,) avg over K controls
            key = (from_lbl, to_lbl)
            transitions.setdefault(key, []).append((probe_win, ctrl_win, tok_at_header))
    return transitions


def plot_leadlag(records: list[dict], out_path: Path, window: int = 12):
    transitions = collect_windows(records, window=window)
    if not transitions:
        print("[leadlag] no forward traces found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(-window, window + 1)

    # ----- Plot 1: per-transition probe trajectory time-locked to header -----
    for (from_lbl, to_lbl), wins in sorted(transitions.items(),
                                            key=lambda kv: PHASE_ORDER.index(kv[0][1])):
        probe_arr = np.array([w[0] for w in wins])
        mean_probe = probe_arr.mean(axis=0)
        sem_probe  = probe_arr.std(axis=0) / max(1, np.sqrt(len(wins)))
        ax1.plot(x, mean_probe, color=PHASE_COLORS[to_lbl], linewidth=2.2,
                 label=f"→ {to_lbl}  (n={len(wins)})")
        ax1.fill_between(x, mean_probe - sem_probe, mean_probe + sem_probe,
                         color=PHASE_COLORS[to_lbl], alpha=0.15)
    # Pooled control
    all_ctrl = np.array([w[1] for ws in transitions.values() for w in ws])
    ax1.fill_between(x, all_ctrl.mean(0) - all_ctrl.std(0),
                     all_ctrl.mean(0) + all_ctrl.std(0),
                     color="gray", alpha=0.20, label="random control ±1σ (pooled)")
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.text(0.2, ax1.get_ylim()[1], "header token", fontsize=8, va="top")
    ax1.axhline(0, color="black", linewidth=0.3, alpha=0.4)
    ax1.set_xlabel("token offset relative to phase header start")
    ax1.set_ylabel(f"probe projection (L{records[0]['layer']})")
    ax1.set_title("Per-token probe trajectory time-locked to phase transition")
    ax1.legend(fontsize=8, loc="best")

    # ----- Plot 2: shift = (post - pre) projection per transition type -----
    # pre  = mean over offsets [-W .. -1]   (last bit of prev body, before header)
    # head = mean over offsets [0 .. ~3]    (first few tokens of header itself)
    # post = mean over offsets [+5 .. +W]   (after header, into new body)
    pre_w  = (-window, 0)
    head_w = (0, 4)
    post_w = (5, window + 1)

    labels, pre_means, head_means, post_means = [], [], [], []
    for (from_lbl, to_lbl), wins in sorted(transitions.items(),
                                            key=lambda kv: PHASE_ORDER.index(kv[0][1])):
        probe_arr = np.array([w[0] for w in wins])  # (n, 2W+1)
        # Slice
        pre_idx  = slice(pre_w[0]  + window, pre_w[1]  + window)
        head_idx = slice(head_w[0] + window, head_w[1] + window)
        post_idx = slice(post_w[0] + window, post_w[1] + window)
        pre_means.append(float(probe_arr[:,  pre_idx].mean()))
        head_means.append(float(probe_arr[:, head_idx].mean()))
        post_means.append(float(probe_arr[:, post_idx].mean()))
        labels.append(f"{from_lbl[:4]}→{to_lbl[:4]}")

    xpos = np.arange(len(labels))
    width = 0.27
    ax2.bar(xpos - width, pre_means,  width, label="pre-header  (offset −W..−1)",
            color="#94a3b8", edgecolor="white")
    ax2.bar(xpos,         head_means, width, label="header  (offset 0..3)",
            color="#475569", edgecolor="white")
    ax2.bar(xpos + width, post_means, width, label="post-header  (offset +5..+W)",
            color="#1F6FEB", edgecolor="white")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.3, alpha=0.4)
    ax2.set_ylabel("mean probe projection")
    ax2.set_title("Probe projection: pre-header vs header vs post-header")
    ax2.legend(fontsize=8, loc="best")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[leadlag] wrote {out_path}")

    # Diagnostic prints
    print("[leadlag] per-transition pre/header/post means:")
    for lbl, pre, hd, post in zip(labels, pre_means, head_means, post_means):
        delta_pre_post = post - pre
        delta_head = hd - pre
        print(f"  {lbl}:  pre={pre:+.2f}  head={hd:+.2f}  post={post:+.2f}  "
              f"Δ(pre→head)={delta_head:+.2f}  Δ(pre→post)={delta_pre_post:+.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=12)
    args = p.parse_args()
    records = load(Path(args.inp))
    print(f"[leadlag] loaded {len(records)} traces from {args.inp}")
    plot_leadlag(records, Path(args.out), window=args.window)


if __name__ == "__main__":
    main()
