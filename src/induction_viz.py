"""
Visualizations for the induction-head experiment.
Outputs:
  - induction_phase.png      loss + induction-score over training (the elbow!)
  - induction_attn_grid.png  attention patterns at 4 moments: before, during, after, final
  - induction_anim.gif       attention of L1.H0 evolving through training
  - induction_compose.png    L0 prev-token + L1 induction, side by side, annotated
"""
import json, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path("/home/stuart/sonnet_mind")
RUN = ROOT / "induction_run"
CKPTS = RUN / "ckpts"
FIGS = ROOT / "run" / "figs"   # write into the main figs directory
FIGS.mkdir(exist_ok=True)

meta = json.loads((RUN / "meta.json").read_text())
log = json.loads((RUN / "log.json").read_text())
P = meta["P"]; SEQ = meta["seq"]
N_LAYERS = meta["n_layers"]; N_HEADS = meta["n_heads"]

plt.rcParams.update({
    "axes.facecolor": "#0e0f12", "figure.facecolor": "#0e0f12",
    "axes.edgecolor": "#444", "axes.labelcolor": "#dcdcdc",
    "axes.titlecolor": "#f5f5f5", "xtick.color": "#aaa", "ytick.color": "#aaa",
    "grid.color": "#2a2c33", "text.color": "#dcdcdc",
    "axes.titlesize": 13, "axes.titleweight": "bold",
})
GOLD = "#e9b44c"; IRIS = "#9a8aff"; MINT = "#43d9ad"; ROSE = "#ff6b8b"

steps = [r["step"] for r in log]
losses = [r["val_loss"] for r in log]
ind_max = [r["ind_max"] for r in log]
ind_per_head = np.array([r["ind_per_head"] for r in log])  # T, L, H

# ------------------------------------------------------------------
# 1. Phase-transition figure — loss and induction together
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(np.array(steps)+1, losses, color=GOLD, lw=2.4, label="val loss (second half)")
ax.scatter(np.array(steps)+1, losses, color=GOLD, s=14, zorder=5)
ax.set_xscale("log")
ax.set_xlabel("training step (log)")
ax.set_ylabel("val loss (nats)", color=GOLD)
ax.tick_params(axis="y", colors=GOLD)
ax.grid(True, alpha=0.3)
# uniform baseline
ax.axhline(math.log(meta["vocab"]), color="#666", ls="--", lw=0.8)
ax.text(1, math.log(meta["vocab"])-0.06,
        f"  uniform-prior ln({meta['vocab']}) = {math.log(meta['vocab']):.2f}",
        color="#888", fontsize=9, va="top")

ax2 = ax.twinx()
# plot induction score per head
for L in range(N_LAYERS):
    for H in range(N_HEADS):
        col = MINT if L == 1 else IRIS
        ls = "-" if L == 1 else ":"
        lw = 2.0 if L == 1 else 1.2
        ax2.plot(np.array(steps)+1, ind_per_head[:, L, H], color=col, ls=ls, lw=lw, alpha=0.85,
                 label=f"L{L}.H{H} induction" if H == 0 else None)
ax2.set_ylabel("induction score", color=MINT)
ax2.tick_params(axis="y", colors=MINT)
ax2.set_ylim(0, 1)

# annotate the phase transition
elbow = next((i for i, l in enumerate(losses) if l < 1.5), None)
if elbow is not None:
    es = steps[elbow]
    ax.axvspan(80, 250, alpha=0.06, color=ROSE, zorder=0)
    ax.text(150, 3.0, "phase\ntransition", color=ROSE, fontsize=11,
            style="italic", ha="center", weight="bold")

# combine legends
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc="upper right", framealpha=0.2, edgecolor="#444", fontsize=9)
ax.set_title("the induction circuit assembling — synthetic 'copy a random prefix' task")
fig.tight_layout()
fig.savefig(FIGS / "induction_phase.png", dpi=150)
plt.close(fig)
print("✓ induction_phase.png")

# ------------------------------------------------------------------
# 2. Attention pattern grid at four moments
# ------------------------------------------------------------------
attn_files = sorted(CKPTS.glob("attn_*.npy"))
attn_steps = [int(f.stem.split("_")[1]) for f in attn_files]
def attn_at(s):
    i = min(range(len(attn_steps)), key=lambda i: abs(attn_steps[i]-s))
    return attn_steps[i], np.load(attn_files[i]).astype(np.float32)

picks = [
    ("before — uniform attention",       attn_at(50)),
    ("during phase transition",          attn_at(140)),
    ("circuit assembled",                attn_at(300)),
    ("converged",                        attn_at(2500)),
]
fig, axes = plt.subplots(N_LAYERS, len(picks)*N_HEADS, figsize=(2.0*len(picks)*N_HEADS, 2.0*N_LAYERS))
for col_block, (title, (s, atn)) in enumerate(picks):
    for L in range(N_LAYERS):
        for H in range(N_HEADS):
            col = col_block * N_HEADS + H
            ax = axes[L, col]
            ax.imshow(atn[L,H], cmap="magma", vmin=0, vmax=min(0.6, atn[L,H].max()))
            if L == 0:
                if H == 0:
                    ax.set_title(f"{title}\nstep {s}\nL0 H0", color="#f5f5f5", fontsize=8.5)
                else:
                    ax.set_title(f"\nL0 H{H}", color="#f5f5f5", fontsize=8.5)
            else:
                ax.set_title(f"L1 H{H}", color="#f5f5f5", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
            # mark the boundary between first half and second half
            ax.axvline(P-0.5, color="#43d9ad", lw=0.5, alpha=0.4)
            ax.axhline(P-0.5, color="#43d9ad", lw=0.5, alpha=0.4)
fig.suptitle("attention patterns through the phase transition  ·  green lines mark boundary between prefix and repeat",
             color="#f5f5f5", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(FIGS / "induction_attn_grid.png", dpi=150)
plt.close(fig)
print("✓ induction_attn_grid.png")

# ------------------------------------------------------------------
# 3. Animation — L1.H0 attention over training
# ------------------------------------------------------------------
keep = list(range(0, len(attn_files), max(1, len(attn_files)//30)))
if keep[-1] != len(attn_files)-1: keep.append(len(attn_files)-1)
keep_files = [attn_files[i] for i in keep]
keep_steps = [attn_steps[i] for i in keep]

fig, axes = plt.subplots(1, N_LAYERS*N_HEADS, figsize=(3.4*N_LAYERS*N_HEADS, 3.6))
ims = []
labels = [f"L{L} H{H}" for L in range(N_LAYERS) for H in range(N_HEADS)]
for ax, lab in zip(axes, labels):
    im = ax.imshow(np.zeros((SEQ-1, SEQ-1)), cmap="magma", vmin=0, vmax=0.5,
                    interpolation="nearest", aspect="auto")
    ax.set_title(lab, color="#f5f5f5"); ax.set_xticks([]); ax.set_yticks([])
    ax.axvline(P-0.5, color="#43d9ad", lw=0.7, alpha=0.5)
    ax.axhline(P-0.5, color="#43d9ad", lw=0.7, alpha=0.5)
    ims.append(im)
title = fig.suptitle("", color="#f5f5f5", fontsize=14, y=0.98)
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.05, wspace=0.1)

def update(frame):
    arr = np.load(keep_files[frame]).astype(np.float32)
    for L in range(N_LAYERS):
        for H in range(N_HEADS):
            ims[L*N_HEADS + H].set_data(arr[L,H])
    title.set_text(f"step {keep_steps[frame]:>5}   ·   the induction circuit forming")
    return ims + [title]

ani = FuncAnimation(fig, update, frames=len(keep_files), interval=180, blit=False)
ani.save(FIGS / "induction_anim.gif", writer=PillowWriter(fps=6), dpi=110)
plt.close(fig)
print("✓ induction_anim.gif")

# ------------------------------------------------------------------
# 4. Composition figure — show L0 (prev-token) and L1 (induction) side by side
# at the converged step, with annotations
# ------------------------------------------------------------------
final_atn = np.load(CKPTS / f"attn_{steps[-1]:06d}.npy").astype(np.float32)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
# Best L0 head — measure prev-token strength
best_L0 = 0
best_score = -1
for H in range(N_HEADS):
    a = final_atn[0, H]
    score = float(np.mean([a[i, i-1] for i in range(1, a.shape[0])]))
    if score > best_score:
        best_score = score; best_L0 = H
# Best L1 head — measure induction strength
best_L1 = int(np.argmax(np.array(log[-1]["ind_per_head"])[1]))

axes[0].imshow(final_atn[0, best_L0], cmap="magma", vmin=0, vmax=0.6)
axes[0].set_title(f"L0.H{best_L0}  —  the previous-token head\n(mean attention to t-1: {best_score:.2f})")
axes[0].axvline(P-0.5, color=MINT, lw=0.8, alpha=0.6)
axes[0].axhline(P-0.5, color=MINT, lw=0.8, alpha=0.6)
axes[0].set_xlabel("attended position"); axes[0].set_ylabel("query position")
# overlay diagonal-1 indicator
T = final_atn.shape[-1]
axes[0].plot([-0.5, T-1.5], [0.5, T-0.5], color="#fff", lw=0.5, ls="--", alpha=0.4)
axes[0].text(T-2, 1, "diagonal −1", color="#fff", fontsize=9, va="top", ha="right",
             rotation=-45, alpha=0.7)

ind_score = np.array(log[-1]["ind_per_head"])[1, best_L1]
axes[1].imshow(final_atn[1, best_L1], cmap="magma", vmin=0, vmax=0.6)
axes[1].set_title(f"L1.H{best_L1}  —  the induction head\n(mean attention to t-P+1 in 2nd half: {ind_score:.2f})")
axes[1].axvline(P-0.5, color=MINT, lw=0.8, alpha=0.6)
axes[1].axhline(P-0.5, color=MINT, lw=0.8, alpha=0.6)
axes[1].set_xlabel("attended position"); axes[1].set_ylabel("query position")
# Annotate the induction line: y=p (in 2nd half), x = p - P + 1 (target lookback)
# That's the line y = x + P - 1 for x in [1, P-1]
xs = np.arange(1, P)
ys = xs + P - 1
axes[1].plot(xs, ys, color="#fff", lw=0.5, ls="--", alpha=0.5)
axes[1].text(P//2, P + P//2 - 1, "induction lookback line\n(y = x + P − 1)",
             color="#fff", fontsize=9, ha="center", va="center", alpha=0.8)

fig.suptitle("the canonical induction circuit  ·  L0 supplies prev-token info, L1 uses it to copy from history",
             color="#f5f5f5", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig(FIGS / "induction_compose.png", dpi=150)
plt.close(fig)
print("✓ induction_compose.png")

print("\nAll induction figs in", FIGS)
