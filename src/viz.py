"""
Visualisations:
  1. loss_curve.png         — val loss vs step (log x), with checkpoint markers
  2. attn_evolution.gif     — attention on probe, evolving across training
  3. heads_final.png        — 4x4 grid of final attention patterns per (layer, head)
  4. fingerprints.png       — heatmap of behavioural metrics per head
  5. induction_bar.png      — induction score per head at final step
  6. logit_lens.png         — bar plot of layer-wise logit-lens prob on probe
  7. generations_grid.png   — visual timeline of sample text at each ckpt
"""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches

ROOT = Path("/home/stuart/sonnet_mind")
RUN = ROOT / "run"
CKPTS = RUN / "ckpts"
ANA = RUN / "analysis"
OUT = RUN / "figs"
OUT.mkdir(exist_ok=True)

meta = json.loads((RUN / "meta.json").read_text())
N_LAYERS = meta["n_layers"]; N_HEADS = meta["n_heads"]; CTX = meta["ctx"]
itos = {int(k):v for k,v in meta["itos"].items()}
PROBE = meta["probe"]

log = json.loads((RUN / "log.json").read_text())
steps = [r["step"] for r in log]
losses = [r["val_loss"] for r in log]
best_idx = int(np.argmin(losses))
best_step = steps[best_idx]
best_loss = losses[best_idx]

# ------------------------------------------------------------------
# Aesthetic
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#0e0f12",
    "figure.facecolor": "#0e0f12",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#dcdcdc",
    "axes.titlecolor": "#f5f5f5",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#2a2c33",
    "text.color": "#dcdcdc",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})
PARCHMENT = "#f4ecd8"
INK = "#1a1a1a"
ACCENT = "#e9b44c"   # warm gold
ACCENT2 = "#9a8aff"  # iris
ACCENT3 = "#43d9ad"  # mint

# ------------------------------------------------------------------
# 1. Loss curve
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(np.array(steps)+1, losses, color=ACCENT, lw=2, alpha=0.9)
ax.scatter(np.array(steps)+1, losses, color=ACCENT2, s=22, zorder=5, edgecolors="#fff", linewidths=0.4)
# mark best
ax.scatter([best_step+1], [best_loss], color=ACCENT3, s=180, zorder=6, marker="*", edgecolors="#fff", linewidths=1, label=f"best (step {best_step}, val={best_loss:.3f})")
ax.axvline(best_step+1, color=ACCENT3, ls=":", lw=0.8, alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("training step (log)")
ax.set_ylabel("validation loss (cross-entropy)")
ax.set_title("the descent into language — and then, the climb back into rote memory")
ax.grid(True, alpha=0.3)
ax.axhline(math.log(meta["vocab"]), color="#666", ls="--", lw=0.8, alpha=0.5)
ax.text(1, math.log(meta["vocab"])-0.06, f"  uniform-prior baseline ln({meta['vocab']}) = {math.log(meta['vocab']):.2f}",
        color="#888", fontsize=9, va="top")
ax.legend(loc="upper right", framealpha=0.2, edgecolor="#444")
fig.tight_layout()
fig.savefig(OUT / "loss_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ loss_curve.png")

# ------------------------------------------------------------------
# 2. Attention evolution GIF — show layer 1 head 0 attending on probe
# Pick the most "interesting" head: choose one with non-trivial attention.
# Show 2x2 grid: top-left head from each layer, evolving across training.
# ------------------------------------------------------------------
attn_files = sorted(CKPTS.glob("attn_*.npy"))
attn_steps = [int(f.stem.split("_")[1]) for f in attn_files]
# Subsample to ~30 frames for a smooth GIF
keep_idx = np.linspace(0, len(attn_files)-1, min(30, len(attn_files))).astype(int)
attn_files_sub = [attn_files[i] for i in keep_idx]
attn_steps_sub = [attn_steps[i] for i in keep_idx]

# choose four heads to show evolution, one per layer (head 0 each)
LH_PAIRS = [(L, 0) for L in range(N_LAYERS)]

fig, axes = plt.subplots(1, N_LAYERS, figsize=(3.4*N_LAYERS, 3.8))
ims = []
for ax, (L,H) in zip(axes, LH_PAIRS):
    im = ax.imshow(np.zeros((CTX, CTX)), cmap="magma", vmin=0, vmax=0.5,
                    interpolation="nearest", aspect="auto")
    ax.set_title(f"L{L} H{H}", color="#f5f5f5")
    ax.set_xticks([]); ax.set_yticks([])
    ims.append(im)
title = fig.suptitle("", color="#f5f5f5", fontsize=14, y=0.98)
fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.05, wspace=0.1)

def update(frame):
    arr = np.load(attn_files_sub[frame]).astype(np.float32)  # L,H,T,T
    for ax, im, (L,H) in zip(axes, ims, LH_PAIRS):
        im.set_data(arr[L,H])
    title.set_text(f"step {attn_steps_sub[frame]:>5}   ·   the eyes open")
    return ims + [title]

ani = FuncAnimation(fig, update, frames=len(attn_files_sub), interval=140, blit=False)
ani.save(OUT / "attn_evolution.gif", writer=PillowWriter(fps=7), dpi=110)
plt.close(fig)
print("✓ attn_evolution.gif")

# ------------------------------------------------------------------
# 3. Best-ckpt attention grid (all heads, all layers)
# ------------------------------------------------------------------
best_attn_file = CKPTS / f"attn_{best_step:06d}.npy"
final_attn = np.load(best_attn_file).astype(np.float32)  # L,H,T,T

fig, axes = plt.subplots(N_LAYERS, N_HEADS, figsize=(2.2*N_HEADS, 2.2*N_LAYERS))
if N_LAYERS == 1: axes = np.array([axes])
if N_HEADS == 1: axes = axes.reshape(-1,1)
for L in range(N_LAYERS):
    for H in range(N_HEADS):
        ax = axes[L, H]
        ax.imshow(final_attn[L,H], cmap="magma", vmin=0, vmax=min(0.6, final_attn[L,H].max()))
        ax.set_title(f"L{L} H{H}", color="#f5f5f5", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"attention patterns at peak generalisation (step {best_step})",
             color="#f5f5f5", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT / "heads_final.png", dpi=150)
plt.close(fig)
print("✓ heads_final.png")

# ------------------------------------------------------------------
# 4. Head fingerprint heatmap
# ------------------------------------------------------------------
fps = json.loads((ANA / "fingerprints.json").read_text())
metrics = ["prev_token", "bos", "current_token", "induction", "sharpness", "entropy"]
M = np.zeros((len(fps), len(metrics)))
for i, f in enumerate(fps):
    for j, m in enumerate(metrics):
        M[i,j] = f[m]

# normalize each column to 0..1 for visual comparison
Mn = (M - M.min(0)) / ((M.max(0) - M.min(0)) + 1e-9)

fig, ax = plt.subplots(figsize=(7, 0.55*len(fps)+1.2))
ax.imshow(Mn, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, rotation=30, ha="right")
ax.set_yticks(range(len(fps)))
ax.set_yticklabels([f"L{f['layer']}.H{f['head']}" for f in fps])
for i in range(len(fps)):
    for j, m in enumerate(metrics):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                color="#fff" if Mn[i,j]<0.6 else "#000", fontsize=9)
ax.set_title("head fingerprints — what does each head do?")
fig.tight_layout()
fig.savefig(OUT / "fingerprints.png", dpi=150)
plt.close(fig)
print("✓ fingerprints.png")

# ------------------------------------------------------------------
# 5. Induction score bar
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.2))
labels = [f"L{f['layer']}.H{f['head']}" for f in fps]
ind = [f["induction"] for f in fps]
colors = [ACCENT if v == max(ind) else (ACCENT2 if v > 0.5*max(ind) else "#555") for v in ind]
bars = ax.bar(labels, ind, color=colors)
# baseline = uniform attention into each position ≈ 1/T over the position; tiny
ax.axhline(1/CTX, color="#666", ls="--", lw=0.8)
ax.text(len(labels)-0.5, 1/CTX, f" chance (1/{CTX}≈{1/CTX:.4f})", color="#888", fontsize=9, va="bottom")
ax.set_title("induction signal — which heads predict 'the next thing after the previous match'?")
ax.set_ylabel("mean attn(t → t-P+1) on repeated random sequence")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "induction_bar.png", dpi=150)
plt.close(fig)
print("✓ induction_bar.png")

# ------------------------------------------------------------------
# 6. Logit lens — show top-5 next-token probabilities at last probe position
# ------------------------------------------------------------------
ll = np.load(ANA / "logit_lens.npy")  # L, T, V
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
probs = softmax(ll[:, -1, :], axis=-1)  # L, V
fig, axes = plt.subplots(1, ll.shape[0], figsize=(2.6*ll.shape[0], 3.2), sharey=True)
for L, ax in enumerate(axes):
    p = probs[L]
    top = np.argsort(p)[::-1][:6]
    chars = [itos[i] if itos[i] != "\n" else "↵" for i in top]
    chars = [" " if c == " " else c for c in chars]
    ax.barh(range(6), p[top][::-1], color=ACCENT)
    ax.set_yticks(range(6))
    ax.set_yticklabels([chars[5-i] for i in range(6)])
    ax.set_title(f"after L{L}")
    ax.set_xlabel("prob")
fig.suptitle("logit lens — what each layer would predict next, given probe", color="#f5f5f5")
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(OUT / "logit_lens.png", dpi=150)
plt.close(fig)
print("✓ logit_lens.png")

# ------------------------------------------------------------------
# 7. Generation timeline as image — show first ~120 chars at log-spaced ckpts
# ------------------------------------------------------------------
sample_files = sorted(CKPTS.glob("sample_*.txt"))
sample_steps = [int(f.stem.split("_")[1]) for f in sample_files]
# pick ~12 evenly log-spaced
idxs = np.unique(np.linspace(0, len(sample_files)-1, 12).astype(int))
chosen = [(sample_steps[i], sample_files[i].read_text()) for i in idxs]

fig, ax = plt.subplots(figsize=(11, 0.85*len(chosen)+1))
ax.axis("off")
fig.patch.set_facecolor(PARCHMENT)
y = 1.0
ax.set_xlim(0,1); ax.set_ylim(0,1)
n = len(chosen)
for i, (step, txt) in enumerate(chosen):
    body = txt[len(PROBE):][:140].replace("\n", "↵")
    yp = 1 - (i+0.5)/n
    ax.text(0.0, yp, f"step {step:>5}", color="#5a4f33", fontsize=10,
            family="monospace", weight="bold", va="center")
    ax.text(0.13, yp, body, color=INK, fontsize=10, family="serif", va="center")
fig.suptitle("the voice emerges — model continuations of the same probe across training",
             color="#3a2f15", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(OUT / "generations_grid.png", dpi=150, facecolor=PARCHMENT)
plt.close(fig)
print("✓ generations_grid.png")

print("\nAll figs in", OUT)
