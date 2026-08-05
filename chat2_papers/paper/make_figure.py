"""
Figure 1 for the URTC paper: source-hospital validation AUROC plotted against
true zero-shot target AUROC, one point per (constraint weight, target site).

The visual argument: validation moves almost not at all along x while target
performance moves an order of magnitude more along y, and the relationship is
downward rather than upward -- so the criterion practitioners actually use
carries no usable signal about deployment performance. The configuration source
validation selects is marked against the best configuration available.

Encoding notes: three sites use the Okabe-Ito colorblind-safe hues AND distinct
marker shapes, so identity survives grayscale printing and CVD. Palette verified
with the dataviz validator (all checks PASS).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig1_validation_gap.png")

# Full-scale constraint-weight sweep, seed 42 (same run reported in Table II).
LAM = ["0.0", "0.1", "0.5", "1.0", "2.0", "5.0"]
VAL = [0.8350, 0.8306, 0.8249, 0.8271, 0.8203, 0.8280]
SITES = {
    "PhysioNet Site B": ([0.6923, 0.7371, 0.7495, 0.6749, 0.6527, 0.5972], "#0072B2", "o"),
    "MIMIC-IV":         ([0.6006, 0.6908, 0.6952, 0.6451, 0.6326, 0.6076], "#E69F00", "s"),
    "eICU-CRD":         ([0.5479, 0.6366, 0.6623, 0.5848, 0.5617, 0.5749], "#009E73", "^"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.linewidth": 0.6,
})

fig, ax = plt.subplots(figsize=(3.45, 2.75), dpi=400)

for label, (ys, color, marker) in SITES.items():
    ax.scatter(VAL, ys, s=26, c=color, marker=marker, label=label,
               edgecolors="white", linewidths=0.5, zorder=3)

# Recessive grid; axes carry the comparison, not decoration.
ax.grid(True, color="#d9d9d9", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#8a8a8a")

ax.set_xlabel("Source-hospital validation AUROC", labelpad=3)
ax.set_ylabel("Zero-shot target AUROC", labelpad=3)

# Make the asymmetry explicit: validation varies ~0.015, target ~0.20.
vspan = max(VAL) - min(VAL)
tall = [v for ys, _, _ in SITES.values() for v in ys]
tspan = max(tall) - min(tall)
ax.annotate("", xy=(min(VAL), 0.535), xytext=(max(VAL), 0.535),
            arrowprops=dict(arrowstyle="<->", color="#5a5a5a", lw=0.7))
ax.text((min(VAL) + max(VAL)) / 2, 0.545,
        f"validation spans {vspan:.3f}", ha="center", va="bottom",
        fontsize=6.5, color="#3a3a3a")
ax.annotate("", xy=(0.8385, min(tall)), xytext=(0.8385, max(tall)),
            arrowprops=dict(arrowstyle="<->", color="#5a5a5a", lw=0.7))
ax.text(0.8375, (min(tall) + max(tall)) / 2, f"target spans {tspan:.2f}",
        rotation=90, ha="right", va="center", fontsize=6.5, color="#3a3a3a")

# The concrete failure: validation's pick vs. the best configuration available.
i_sel = VAL.index(max(VAL))                       # source validation selects this
i_best = SITES["eICU-CRD"][0].index(max(SITES["eICU-CRD"][0]))
ax.annotate(f"selected by\nsource validation\n(weight {LAM[i_sel]})",
            xy=(VAL[i_sel], SITES["eICU-CRD"][0][i_sel]),
            xytext=(VAL[i_sel] - 0.001, 0.575), fontsize=6.2, color="#3a3a3a",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#3a3a3a", lw=0.6))
ax.annotate(f"best available\n(weight {LAM[i_best]})",
            xy=(VAL[i_best], SITES["eICU-CRD"][0][i_best]),
            xytext=(VAL[i_best] - 0.0075, 0.695), fontsize=6.2, color="#3a3a3a",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#3a3a3a", lw=0.6))

ax.set_xlim(0.8155, 0.8405)
ax.set_ylim(0.525, 0.775)
ax.legend(frameon=False, fontsize=6.5, loc="upper left",
          handletextpad=0.35, borderaxespad=0.2, labelspacing=0.28)

fig.tight_layout(pad=0.35)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: study design. Which site is used for which decision -- the
# reviewers of the earlier draft asked specifically for this schematic.
# ══════════════════════════════════════════════════════════════════════════
def figure2():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(3.4, 1.45), dpi=400)
    ax.set_xlim(0, 10); ax.set_ylim(0.0, 5.3); ax.axis("off")

    def box(x, y, w, h, title, sub, face, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    facecolor=face, edgecolor=edge, linewidth=0.8))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=6.8, weight="bold", color="#1a1a1a")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=5.9, color="#3a3a3a")

    box(0.1, 3.55, 4.3, 1.45, "PhysioNet 2019 Site A", "train + held-out validation",
        "#dbeafe", "#0072B2")
    box(5.6, 3.55, 4.3, 1.45, "PhysioNet 2019 Site B", "same collection protocol",
        "#fdf0d5", "#E69F00")
    box(0.1, 1.55, 4.3, 1.45, "MIMIC-IV  (74,607)", "independent EHR system",
        "#d9f2e6", "#009E73")
    box(5.6, 1.55, 4.3, 1.45, "eICU-CRD  (130,446)", "many hospitals",
        "#d9f2e6", "#009E73")

    for xy, xytext in [((5.55, 4.25), (4.45, 4.25)),
                       ((2.25, 3.05), (2.25, 3.5)),
                       ((7.75, 3.05), (7.75, 3.5))]:
        ax.add_patch(FancyArrowPatch(xytext, xy, arrowstyle="->", mutation_scale=7,
                                     color="#5a5a5a", linewidth=0.8))
    ax.text(5.0, 4.5, "zero-shot", ha="center", fontsize=5.6, color="#3a3a3a")
    ax.text(0.1, 0.75, "Model selection uses Site A validation or unlabeled target "
                       "inputs only.", fontsize=5.9, color="#3a3a3a")
    ax.text(0.1, 0.2, "Target labels are used for final reporting, never for "
                       "any selection decision.", fontsize=5.9, color="#3a3a3a")
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(HERE, "fig2_design.png"), bbox_inches="tight",
                facecolor="white")
    print("wrote fig2_design.png")


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: selection regret by criterion. Magnitude comparison -> bars.
# Grey is the field of candidates; the two bars that carry the argument
# (standard practice vs. the proposed substitute) are the only ones colored.
# ══════════════════════════════════════════════════════════════════════════
def figure3():
    items = [("Predictive entropy", -0.076, "#b9b9b9"),
             ("Source validation", -0.052, "#E69F00"),
             ("Random selection", -0.043, "#b9b9b9"),
             ("Representation distance", -0.030, "#b9b9b9"),
             ("Constraint violation", -0.016, "#b9b9b9"),
             ("Reconstruction error", -0.012, "#0072B2")]
    fig, ax = plt.subplots(figsize=(3.4, 1.95), dpi=400)
    ys = range(len(items))
    ax.barh(list(ys), [abs(v) for _, v, _ in items],
            color=[c for _, _, c in items], height=0.62, zorder=3)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([n for n, _, _ in items], fontsize=6.6)
    for i, (_, v, _) in enumerate(items):
        ax.text(abs(v) + 0.0016, i, f"{abs(v):.3f}", va="center",
                fontsize=6.2, color="#3a3a3a")
    ax.set_xlabel("Mean selection regret (AUROC below best available)",
                  fontsize=6.8, labelpad=2)
    ax.tick_params(axis="x", labelsize=6.2)
    ax.grid(True, axis="x", color="#d9d9d9", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#8a8a8a")
    ax.set_xlim(0, 0.092)
    fig.tight_layout(pad=0.25)
    fig.savefig(os.path.join(HERE, "fig3_regret.png"), bbox_inches="tight",
                facecolor="white")
    print("wrote fig3_regret.png")


figure2()
figure3()
