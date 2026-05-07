"""
Mechanistic interpretability of the trained tiny GPT.
Loads the final checkpoint and runs:
  1. Per-head attention pattern characterization (previous-token, current-token, BOS, induction)
  2. Induction head detection on repeated random sequences
  3. Logit-lens style analysis of probe completions
  4. Attention head specialization heatmap
Saves analysis artifacts under run/analysis/.
"""
import os, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path("/home/stuart/sonnet_mind")
RUN  = ROOT / "run"
CKPTS = RUN / "ckpts"
ANA = RUN / "analysis"
ANA.mkdir(parents=True, exist_ok=True)

meta = json.loads((RUN / "meta.json").read_text())
VOCAB    = meta["vocab"]
CTX      = meta["ctx"]
D_MODEL  = meta["d_model"]
N_HEADS  = meta["n_heads"]
N_LAYERS = meta["n_layers"]
stoi     = meta["stoi"]
itos     = {int(k):v for k,v in meta["itos"].items()}
PROBE    = meta["probe"]
DEVICE = "cuda"

def encode(s): return [stoi[c] for c in s]
def decode(t): return "".join(itos[int(i)] for i in t)

# --- model (same arch, must match train.py) ---
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3*D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
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
            x = x + a; x = x + self.mlp(self.ln2(x))
            return x, attn
        x = x + self.attn(self.ln1(x)); x = x + self.mlp(self.ln2(x))
        return x

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

# --- pick best-val-loss ckpt (NOT necessarily the last; we expect overfitting) ---
log = json.loads((RUN / "log.json").read_text())
best = min(log, key=lambda r: r["val_loss"])
ckpt_path = CKPTS / f"ckpt_{best['step']:06d}.pt"
print(f"Loading BEST ckpt: step={best['step']} val_loss={best['val_loss']:.4f}")
state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
model = TinyGPT().to(DEVICE).eval()
model.load_state_dict(state["model"])
# Save which ckpt we used
(ANA / "best_ckpt.json").write_text(json.dumps({
    "step": best["step"], "val_loss": best["val_loss"],
    "last_step": log[-1]["step"], "last_val_loss": log[-1]["val_loss"],
}, indent=2))

# --------------------------------------------------------------------
# 1. Per-head behavioral fingerprint
# Run model on a stretch of real Shakespeare text and characterize attention
# --------------------------------------------------------------------
text = (ROOT / "shakespeare.txt").read_text()
sample = text[1000:1000+CTX]
ids = torch.tensor([encode(sample)], device=DEVICE)

with torch.no_grad():
    logits, attns = model(ids, return_attn=True)
# attns: [L][1,H,T,T]
attn_arr = np.stack([a[0].cpu().numpy() for a in attns])  # L,H,T,T
np.save(ANA / "attn_real_text.npy", attn_arr.astype(np.float32))

# Behavioral metrics per head:
#  - prev_token_score: avg attention on diagonal -1 (excluding pos 0)
#  - current_token_score: avg attention on diagonal 0 (effectively no-op via residual)
#  - bos_score: avg attention on token 0 (excluding pos 0)
#  - entropy (avg over rows): high = diffuse, low = sharp
T = attn_arr.shape[-1]
fingerprints = []
for L in range(N_LAYERS):
    for H in range(N_HEADS):
        a = attn_arr[L,H]  # T,T
        rows = np.arange(1, T)
        prev = float(np.mean([a[i, i-1] for i in rows]))
        curr = float(np.mean([a[i, i]   for i in rows]))
        bos  = float(np.mean([a[i, 0]   for i in rows]))
        # row entropy
        rA = np.clip(a[rows,:rows[-1]+1], 1e-12, None)
        ent = float(np.mean([-np.sum(rA[i]*np.log(rA[i])) for i in range(len(rows))]))
        # sharpness: avg of max attention per row
        sharp = float(np.mean([a[i, :i+1].max() for i in rows]))
        fingerprints.append({
            "layer": L, "head": H,
            "prev_token": prev, "current_token": curr, "bos": bos,
            "entropy": ent, "sharpness": sharp,
        })

# --------------------------------------------------------------------
# 2. Induction-head test
# Build a sequence of two repeated random-ish chunks: [A | B | A | ...]
# At positions inside the SECOND A, an induction head should attend to
# the position right after the matching prefix in the FIRST A (i.e. it
# can predict the next token from the first occurrence).
# Score = mean attention from token at position (P + i) to position (i + 1),
# where P is the prefix length.
# --------------------------------------------------------------------
rng = np.random.default_rng(0)
P = 80  # prefix length
suffix = ""  # nothing between, just direct repeat
prefix_ids = rng.integers(0, VOCAB, size=P).tolist()
seq = prefix_ids + prefix_ids  # length 2P
seq = seq[:CTX]
ids = torch.tensor([seq], device=DEVICE)
with torch.no_grad():
    _, attns2 = model(ids, return_attn=True)
attn2 = np.stack([a[0].cpu().numpy() for a in attns2])  # L,H,T,T

induction_scores = np.zeros((N_LAYERS, N_HEADS), dtype=np.float64)
# At position p in the second copy (p in [P+1, 2P-1]), induction head should
# attend to position (p - P + 1) which is the token that came right after
# the matching prefix in the first copy.
for L in range(N_LAYERS):
    for H in range(N_HEADS):
        a = attn2[L,H]
        scores = []
        for p in range(P+1, 2*P):
            target = p - P + 1
            scores.append(a[p, target])
        induction_scores[L,H] = float(np.mean(scores))
# Also do an averaged version with multiple random prefixes for robustness
n_trials = 8
ind_acc = np.zeros((N_LAYERS, N_HEADS), dtype=np.float64)
for trial in range(n_trials):
    pref = rng.integers(0, VOCAB, size=P).tolist()
    seq = (pref + pref)[:CTX]
    ids = torch.tensor([seq], device=DEVICE)
    with torch.no_grad():
        _, attns2 = model(ids, return_attn=True)
    a2 = np.stack([a[0].cpu().numpy() for a in attns2])
    for L in range(N_LAYERS):
        for H in range(N_HEADS):
            a = a2[L,H]
            ind_acc[L,H] += np.mean([a[p, p - P + 1] for p in range(P+1, 2*P)]) / n_trials

for f in fingerprints:
    f["induction"] = float(ind_acc[f["layer"], f["head"]])

# Save fingerprints
(ANA / "fingerprints.json").write_text(json.dumps(fingerprints, indent=2))
print("\nHead fingerprints (final model):")
print(f"{'L.H':>4} | {'prev':>6} {'bos':>6} {'curr':>6} {'sharp':>6} {'ent':>6} {'IND':>6}")
for f in fingerprints:
    print(f"{f['layer']}.{f['head']:>2} | {f['prev_token']:6.3f} {f['bos']:6.3f} "
          f"{f['current_token']:6.3f} {f['sharpness']:6.3f} {f['entropy']:6.3f} {f['induction']:6.3f}")

# --------------------------------------------------------------------
# 3. Logit lens — what does each layer "want to predict" at each position?
# Apply the unembed (head + lnf) to residual stream after each block.
# --------------------------------------------------------------------
# Re-run with hooks to capture residual after each block
layer_residuals = []
def hook(mod, inp, out):
    layer_residuals.append(out.detach() if isinstance(out, torch.Tensor) else out[0].detach())
hooks = []
for b in model.blocks:
    hooks.append(b.register_forward_hook(hook))

probe_text = PROBE
ids = torch.tensor([encode(probe_text)[:CTX]], device=DEVICE)
layer_residuals.clear()
with torch.no_grad():
    _ = model(ids)
for h in hooks: h.remove()

# Apply lnf+head to each
ll = []
with torch.no_grad():
    for r in layer_residuals:
        x = model.lnf(r)
        logits_per_layer = model.head(x)
        ll.append(logits_per_layer[0].detach().cpu().numpy())  # T,V
np.save(ANA / "logit_lens.npy", np.stack(ll).astype(np.float32))

# top-1 next-token prediction per layer
print("\nLogit lens — top-1 prediction at last position per layer:")
for L, lll in enumerate(ll):
    top = int(lll[-1].argmax())
    print(f"  Layer {L}: '{itos[top]}' (logit={lll[-1, top]:.2f})")

# --------------------------------------------------------------------
# 4. Save final attention patterns for plotting
# --------------------------------------------------------------------
# Also save attention on Shakespeare probe at the BEST ckpt (self-contained)
best_attn = np.load(CKPTS / f"attn_{best['step']:06d}.npy").astype(np.float32)
np.save(ANA / "attn_probe_best.npy", best_attn)

print("\nDone. Artifacts in", ANA)
