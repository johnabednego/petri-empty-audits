"""Figures for the write-up, drawn from results/ only."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(os.path.dirname(__file__), "..", "figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9993"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def title(fig, main, sub):
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.12 / h, main, fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 1 - 0.42 / h, sub, fontsize=10.5, color=INK2, va="top")


# Figure 1: share of audits with no conversation, per organism, by configuration
org = pd.read_csv(f"{R}/empty_by_organism.csv")
order = ["Synthetic documents + KTO", "Transcripts + KTO", "Transcripts + SFT", "Synthetic documents + SFT"]
fig, ax = plt.subplots(figsize=(8.6, 3.9))
fig.subplots_adjust(left=0.27, right=0.97, top=0.76, bottom=0.17)
rng = np.random.default_rng(0)
for i, cfg in enumerate(order):
    v = org[org.config == cfg].pct_empty.values
    c = ORANGE if i == 0 else BLUE
    ax.scatter(v, i + rng.uniform(-0.16, 0.16, len(v)), s=46, color=c, alpha=0.9, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    m = np.median(v)
    ax.plot([m, m], [i - 0.3, i + 0.3], color=INK, lw=2, zorder=4)
    ax.text(m, i - 0.36, f"median {m:.0f}%", ha="center", va="bottom", fontsize=9, color=INK2)
ax.set_yticks(range(4), order, color=INK)
ax.invert_yaxis()
ax.set_xlim(10, 40)
ax.set_xticks([10, 15, 20, 25, 30, 35, 40], [f"{t}%" for t in [10, 15, 20, 25, 30, 35, 40]])
ax.set_xlabel("Audits with no conversation, per model organism (14 organisms per row)")
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax.tick_params(axis="y", length=0)
title(fig, "One training configuration has a third of its audits empty",
      "4,065 of 17,808 AuditBench Petri audits contain no conversation, yet each is scored as 'not concerning'")
fig.savefig(f"{F}/fig1_empty_audits_by_configuration.png", dpi=200)
plt.close(fig)

# Figure 2: KTO minus SFT on 'concerning', keep all vs drop empty, per stratum
v = pd.read_csv(f"{R}/kto_vs_sft_by_rule.csv")
v = v[v.score == "concerning"]
w = v.pivot_table(index=["behaviour", "data"], columns="rule", values="diff").reset_index()
q = v.pivot_table(index=["behaviour", "data"], columns="rule", values="q_bh").reset_index()
ka, de = "Keep all audits", "Drop all empty audits"
w = w.sort_values(de).reset_index(drop=True)
q = q.set_index(["behaviour", "data"]).loc[list(zip(w.behaviour, w.data))].reset_index()
labels = [f"{b.replace('_', ' ')} · {'docs' if dt == 'synth_docs' else 'transcripts'}" for b, dt in zip(w.behaviour, w.data)]
fig, ax = plt.subplots(figsize=(8.6, 9.2))
fig.subplots_adjust(left=0.36, right=0.96, top=0.915, bottom=0.075)
y = np.arange(len(w))
flip = np.sign(w[ka]) != np.sign(w[de])
for i in y:
    ax.plot([w[ka][i], w[de][i]], [i, i], color=INK if flip[i] else GRID, lw=1.6 if flip[i] else 2, zorder=1)
ax.scatter(w[ka], y, s=52, color=GREY, edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Keep all audits")
ax.scatter(w[de], y, s=52, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3, label="Drop audits with no conversation")
for i in y:
    if q[de][i] < 0.05:
        ax.text(w[de][i] + (0.07 if w[de][i] >= w[ka][i] else -0.07), i, "*", color=BLUE, fontsize=13, va="center",
                ha="left" if w[de][i] >= w[ka][i] else "right", fontweight="bold")
ax.axvline(0, color=INK2, lw=1, zorder=0)
ax.set_yticks(y, labels, fontsize=9.5, color=INK)
for t, f in zip(ax.get_yticklabels(), flip):
    if f:
        t.set_fontweight("bold")
ax.set_ylim(-0.8, len(w) - 0.2)
ax.set_xlabel("KTO minus SFT, mean 'concerning' score (1 to 10 scale)")
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower right", frameon=False, fontsize=9.5)
ax.text(0.99, 0.12, "* significant after dropping (BH q < .05)\nBold rows: sign flips between rules",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK2)
title(fig, "Dropping empty audits flips the sign in 5 of 28 comparisons",
      "Effect of KTO vs SFT adversarial training on Petri's 'concerning' score, per behaviour and training data")
fig.savefig(f"{F}/fig2_kto_vs_sft_by_rule.png", dpi=200)
plt.close(fig)

# Figure 3: how often a subsample of k audits per organism reproduces a significant difference
s = pd.read_csv(f"{R}/subsample_summary.csv")
fig, ax = plt.subplots(figsize=(8.6, 4.2))
fig.subplots_adjust(left=0.14, right=0.8, top=0.78, bottom=0.15)
for rule, c, lab in [("Drop all empty audits", BLUE, "Empty audits dropped"), ("Keep all audits", GREY, "All audits kept")]:
    t = s[s.rule == rule]
    ax.plot(t.k, 100 * t.median_replication, color=c, lw=2, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
    last = t.iloc[-1]
    ax.text(last.k + 4, 100 * last.median_replication, lab, color=INK, va="center", fontsize=9.5)
    for _, r in t.iterrows():
        if rule == "Drop all empty audits":
            ax.text(r.k, 100 * r.median_replication + 4.5, f"{100 * r.median_replication:.0f}%", ha="center", fontsize=9, color=INK2)
ax.set_xticks([10, 25, 50, 100, 150])
ax.set_xlim(0, 160)
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Audits per model organism")
ax.set_ylabel("Subsamples reproducing the\nfull-data result (median stratum)")
ax.grid(axis="y", color=GRID, lw=0.8)
title(fig, "Small audit sets rarely reproduce the full-data result",
      "Strata where KTO and SFT differ significantly in the full data; 300 random subsamples per size")
fig.savefig(f"{F}/fig3_audits_needed.png", dpi=200)
plt.close(fig)
print("figures written")
