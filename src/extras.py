"""
Bonus analyses:
  - char_embed_pca.png   : token embeddings projected to 2D, colored by category
  - pos_embed_sim.png    : cosine-sim matrix of position embeddings
  - head_deep_dive.png   : for the most "previous-token-like" head in L1, what
                           does its attention look like as colored chars?
  - induction_bar (fix)  : redraw with non-overlapping labels
  - logit_lens (fix)     : redraw with readable space label
  - generations_grid_v2  : wider, more elegant version
"""
import json, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/stuart/sonnet_mind")
RUN = ROOT / "run"
ANA = RUN / "analysis"
FIGS = RUN / "figs"
CKPTS = RUN / "ckpts"

meta = json.loads((RUN / "meta.json").read_text())
log = json.loads((RUN / "log.json").read_text())
fps = json.loads((ANA / "fingerprints.json").read_text())
best = json.loads((ANA / "best_ckpt.json").read_text())

VOCAB = meta["vocab"]; CTX = meta["ctx"]; D_MODEL = meta["d_model"]
N_HEADS = meta["n_heads"]; N_LAYERS = meta["n_layers"]
itos = {int(k):v for k,v in meta["itos"].items()}
stoi = meta["stoi"]
PROBE = meta["probe"]

plt.rcParams.update({
    "axes.facecolor": "#0e0f12", "figure.facecolor": "#0e0f12",
    "axes.edgecolor": "#444", "axes.labelcolor": "#dcdcdc",
    "axes.titlecolor": "#f5f5f5", "xtick.color": "#aaa", "ytick.color": "#aaa",
    "grid.color": "#2a2c33", "text.color": "#dcdcdc",
    "axes.titlesize": 13, "axes.titleweight": "bold",
})
ACCENT = "#e9b44c"; ACCENT2 = "#9a8aff"; ACCENT3 = "#43d9ad"; ROSE = "#ff6b8b"

# Load best ckpt's embeddings (need full state)
state = torch.load(CKPTS / f"ckpt_{best['step']:06d}.pt", map_location="cpu", weights_only=False)
sd = state["model"]
tok_emb = sd["tok.weight"].cpu().numpy()  # V, D
pos_emb = sd["pos.weight"].cpu().numpy()  # T, D

# ------------------------------------------------------------------
# Character embedding PCA — what does the model think characters ARE?
# ------------------------------------------------------------------
# Center
X = tok_emb - tok_emb.mean(axis=0, keepdims=True)
# PCA via SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)
proj = U[:, :2] * S[:2]
expl = (S[:2]**2) / (S**2).sum()

def cat_of(c):
    if c == " ": return "space"
    if c == "\n": return "newline"
    if c.isdigit(): return "digit"
    if c.isupper(): return "uppercase"
    if c.islower(): return "lowercase"
    return "punct"
cats = [cat_of(itos[i]) for i in range(VOCAB)]
cat_color = {
    "lowercase": "#9a8aff", "uppercase": "#e9b44c",
    "digit": "#ff6b8b", "punct": "#43d9ad",
    "space": "#fafafa", "newline": "#fafafa",
}

fig, ax = plt.subplots(figsize=(9, 7))
seen = set()
for i in range(VOCAB):
    c = itos[i]
    cat = cats[i]
    label = cat if cat not in seen else None
    seen.add(cat)
    ax.scatter(proj[i,0], proj[i,1], color=cat_color[cat], s=180, alpha=0.7,
               edgecolors="#222", linewidths=0.5, label=label, zorder=3)
    text = c if c not in (" ", "\n") else ("·" if c==" " else "↵")
    ax.text(proj[i,0], proj[i,1], text, ha="center", va="center",
            color="#1a1a1a", fontsize=10, weight="bold", zorder=4,
            family="monospace")
ax.set_title(f"character embedding PCA  ·  components 1+2 explain {(expl[0]+expl[1])*100:.0f}% var")
ax.set_xlabel(f"PC1 ({expl[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({expl[1]*100:.1f}%)")
ax.grid(True, alpha=0.2)
ax.legend(loc="upper right", fontsize=9, framealpha=0.2, edgecolor="#444")
fig.tight_layout()
fig.savefig(FIGS / "char_embed_pca.png", dpi=150)
plt.close(fig)
print("✓ char_embed_pca.png")

# ------------------------------------------------------------------
# Position-embedding cosine similarity matrix
# ------------------------------------------------------------------
P = pos_emb / (np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-9)
sim = P @ P.T  # T, T
# 2D PCA of position embeddings to show the "ribbon"
Pc = pos_emb - pos_emb.mean(axis=0, keepdims=True)
Up, Sp, _ = np.linalg.svd(Pc, full_matrices=False)
proj = Up[:, :2] * Sp[:2]
expl = (Sp[:2]**2) / (Sp**2).sum()

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
im = axes[0].imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_title(f"position-embedding cosine similarity  ·  {CTX}×{CTX}")
axes[0].set_xlabel("position j"); axes[0].set_ylabel("position i")
plt.colorbar(im, ax=axes[0], fraction=0.046)

# colored 2D ribbon
sc = axes[1].scatter(proj[:,0], proj[:,1], c=np.arange(CTX), cmap="viridis", s=14, alpha=0.85)
axes[1].plot(proj[:,0], proj[:,1], color="#444", lw=0.6, alpha=0.6, zorder=1)
axes[1].set_title(f"position-embedding 2D PCA  ·  PC1+PC2 = {(expl[0]+expl[1])*100:.0f}% var")
axes[1].set_xlabel(f"PC1 ({expl[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({expl[1]*100:.1f}%)")
axes[1].grid(True, alpha=0.2)
cbar = plt.colorbar(sc, ax=axes[1], fraction=0.046)
cbar.set_label("position index", color="#dcdcdc")
fig.suptitle("how the model represents 'where am I in the sentence?'", color="#f5f5f5")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(FIGS / "pos_embed_sim.png", dpi=150)
plt.close(fig)
print("✓ pos_embed_sim.png (with PCA ribbon)")

# ------------------------------------------------------------------
# Head deep-dive: pick the cleanest "previous-token" head and the cleanest
# induction-ish head, plot their attention on a hand-picked Shakespeare line
# with the source/destination tokens shown explicitly.
# ------------------------------------------------------------------
# Need to actually run the model to get attention on a hand-picked snippet
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3*D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.n_heads = N_HEADS; self.head_dim = D_MODEL // N_HEADS
        self.register_buffer("mask", torch.tril(torch.ones(CTX, CTX)).view(1,1,CTX,CTX))
    def forward(self, x, return_attn=False):
        B,T,C = x.size()
        q,k,v = self.qkv(x).split(C, dim=2)
        q = q.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        y = self.proj(y)
        return (y, att) if return_attn else y
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL); self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4*D_MODEL), nn.GELU(), nn.Linear(4*D_MODEL, D_MODEL))
    def forward(self, x, return_attn=False):
        if return_attn:
            a, attn = self.attn(self.ln1(x), return_attn=True)
            return x + a + self.mlp(self.ln2(x + a)), attn
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL); self.pos = nn.Embedding(CTX, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.lnf = nn.LayerNorm(D_MODEL); self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.tok.weight
    def forward(self, idx, return_attn=False):
        B,T = idx.size()
        pos = torch.arange(0,T,device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        attns=[]
        for b in self.blocks:
            if return_attn: x, a = b(x, return_attn=True); attns.append(a)
            else: x = b(x)
        x = self.lnf(x); logits = self.head(x)
        return (logits, attns) if return_attn else logits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyGPT().to(DEVICE).eval()
model.load_state_dict(sd)

# Use a clean, short, recognizable Shakespeare line
snippet = "ROMEO:\nO Romeo, Romeo, wherefore art thou Romeo?\n"
ids = torch.tensor([[stoi[c] for c in snippet]], device=DEVICE)
with torch.no_grad():
    _, atts = model(ids, return_attn=True)
attn = np.stack([a[0].cpu().numpy() for a in atts])  # L,H,T,T

# Pick: cleanest prev-token head and cleanest induction head
prev_top  = max(fps, key=lambda f: f["prev_token"])
ind_top   = max(fps, key=lambda f: f["induction"])

def annotated_heatmap(ax, A, tokens, title):
    T = len(tokens)
    im = ax.imshow(A[:T,:T], cmap="magma", vmin=0, vmax=min(0.7, A[:T,:T].max()))
    # token labels
    labels = [c if c not in (" ","\n") else ("·" if c==" " else "↵") for c in tokens]
    ax.set_xticks(range(T)); ax.set_yticks(range(T))
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("attended (key) →")
    ax.set_ylabel("← from (query)")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
annotated_heatmap(axes[0], attn[prev_top["layer"], prev_top["head"]], snippet,
                  f"L{prev_top['layer']}.H{prev_top['head']} — strongest previous-token head")
annotated_heatmap(axes[1], attn[ind_top["layer"], ind_top["head"]], snippet,
                  f"L{ind_top['layer']}.H{ind_top['head']} — strongest induction-like head")
fig.suptitle("two heads, two algorithms — annotated by character", color="#f5f5f5", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(FIGS / "head_deep_dive.png", dpi=150)
plt.close(fig)
print("✓ head_deep_dive.png")

# ------------------------------------------------------------------
# Fix induction bar — use angled labels and proper layout
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4))
labels = [f"L{f['layer']}.H{f['head']}" for f in fps]
ind = [f["induction"] for f in fps]
maxv = max(ind)
colors = [ACCENT if v == maxv else (ACCENT2 if v > 0.5*maxv else "#555") for v in ind]
bars = ax.bar(range(len(labels)), ind, color=colors)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
ax.axhline(1/CTX, color="#666", ls="--", lw=0.8)
ax.text(len(labels)-0.5, 1/CTX, f" chance (1/{CTX}≈{1/CTX:.4f})", color="#888", fontsize=9, va="bottom")
ax.set_title("induction-like signal across heads — repeats sequence test")
ax.set_ylabel("mean attn(t → t-P+1)")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS / "induction_bar.png", dpi=150)
plt.close(fig)
print("✓ induction_bar.png (fixed)")

# ------------------------------------------------------------------
# Fix logit lens — replace space char in labels
# ------------------------------------------------------------------
ll = np.load(ANA / "logit_lens.npy")  # L, T, V
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)
probs = softmax(ll[:, -1, :], axis=-1)
fig, axes = plt.subplots(1, ll.shape[0], figsize=(2.6*ll.shape[0], 3.6), sharey=True)
for L, ax in enumerate(axes):
    p = probs[L]
    top = np.argsort(p)[::-1][:6]
    chars = []
    for i in top:
        c = itos[i]
        if c == " ": chars.append("'·'")
        elif c == "\n": chars.append("'↵'")
        else: chars.append(c)
    ax.barh(range(6), p[top][::-1], color=ACCENT)
    ax.set_yticks(range(6))
    ax.set_yticklabels([chars[5-i] for i in range(6)], fontsize=11, family="monospace")
    ax.set_title(f"after L{L}", fontsize=12)
    ax.set_xlabel("prob")
    ax.grid(True, axis="x", alpha=0.2)
fig.suptitle("logit lens — top-6 next-char predictions, by layer (probe ends '...burn bright')",
             color="#f5f5f5", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.92])
fig.savefig(FIGS / "logit_lens.png", dpi=150)
plt.close(fig)
print("✓ logit_lens.png (fixed)")

# ------------------------------------------------------------------
# Generations grid v2 — wider, prettier
# ------------------------------------------------------------------
sample_files = sorted(CKPTS.glob("sample_*.txt"))
sample_steps = [int(f.stem.split("_")[1]) for f in sample_files]
idxs = np.unique(np.linspace(0, len(sample_files)-1, 12).astype(int))
chosen = [(sample_steps[i], sample_files[i].read_text()) for i in idxs]

PARCHMENT = "#f4ecd8"
INK = "#1a1a1a"
fig, ax = plt.subplots(figsize=(14, 0.95*len(chosen)+1))
ax.axis("off")
fig.patch.set_facecolor(PARCHMENT)
ax.set_xlim(0,1); ax.set_ylim(0,1)
n = len(chosen)
for i, (step, txt) in enumerate(chosen):
    body = txt[len(PROBE):][:170].replace("\n", "↵")
    yp = 1 - (i+0.5)/n
    ax.text(0.0, yp, f"step {step:>5}", color="#5a4f33", fontsize=10,
            family="monospace", weight="bold", va="center")
    ax.text(0.10, yp, body, color=INK, fontsize=10.5, family="serif", va="center")
fig.suptitle("the voice emerges — same probe, different moments in training",
             color="#3a2f15", fontsize=14, weight="bold", style="italic")
fig.tight_layout()
fig.savefig(FIGS / "generations_grid.png", dpi=150, facecolor=PARCHMENT)
plt.close(fig)
print("✓ generations_grid.png (v2)")

print("\nAll bonus figs in", FIGS)
