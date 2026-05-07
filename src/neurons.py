"""
Neuron viewer for the Shakespeare model.

For every MLP neuron in every layer of the trained model, find the
top-K (token, context) windows from the corpus that maximally activate
that neuron. Then filter to "specialized" neurons whose activation
distribution is most concentrated (kurtotic) — those are the candidates
for interpretability.

Output: run/analysis/neurons.json — a list of dicts, one per chosen
neuron, each with its layer/index/peak-activation-value/top-k snippets
with character-level activation values for highlighting.
"""
import os, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/home/stuart/sonnet_mind")
RUN  = ROOT / "run"
CKPTS = RUN / "ckpts"
ANA  = RUN / "analysis"
ANA.mkdir(parents=True, exist_ok=True)

meta = json.loads((RUN / "meta.json").read_text())
best = json.loads((ANA / "best_ckpt.json").read_text())
VOCAB    = meta["vocab"]
CTX      = meta["ctx"]
D_MODEL  = meta["d_model"]
N_HEADS  = meta["n_heads"]
N_LAYERS = meta["n_layers"]
DEVICE   = "cuda"
itos = {int(k):v for k,v in meta["itos"].items()}
stoi = meta["stoi"]
D_MLP = 4 * D_MODEL

# ---------- model (same arch as training) ----------
class CSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3*D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.n_heads = N_HEADS
        self.head_dim = D_MODEL // N_HEADS
        self.register_buffer("mask", torch.tril(torch.ones(CTX, CTX)).view(1,1,CTX,CTX))
    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.qkv(x).split(C, dim=2)
        q = q.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL); self.attn = CSA()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, 4*D_MODEL)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4*D_MODEL, D_MODEL)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        # Expose post-GELU MLP activations
        h = self.fc1(self.ln2(x))
        a = self.act(h)
        x = x + self.fc2(a)
        return x, a

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL); self.pos = nn.Embedding(CTX, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.lnf = nn.LayerNorm(D_MODEL); self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.tok.weight
    def forward(self, idx):
        B,T = idx.size()
        pos = torch.arange(0,T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        acts = []
        for b in self.blocks:
            x, a = b(x)
            acts.append(a)
        x = self.lnf(x); logits = self.head(x)
        return logits, acts

# ---------- load best ckpt — but the train.py block uses an nn.Sequential MLP,
# whose state-dict keys differ from this module's. Translate.
state = torch.load(CKPTS / f"ckpt_{best['step']:06d}.pt",
                   map_location=DEVICE, weights_only=False)
sd_train = state["model"]
sd = {}
for k, v in sd_train.items():
    nk = k
    # nn.Sequential indices: blocks.{i}.mlp.0.weight -> blocks.{i}.fc1.weight
    nk = nk.replace(".mlp.0.", ".fc1.")
    nk = nk.replace(".mlp.2.", ".fc2.")
    sd[nk] = v

model = TinyGPT().to(DEVICE).eval()
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"Loaded ckpt — missing: {len(missing)}, unexpected: {len(unexpected)}")
print(f"From step {state['step']} val_loss {state['val_loss']:.4f}")

# ---------- collect activations on Shakespeare ----------
text = (ROOT / "shakespeare.txt").read_text()
ids = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=DEVICE)
print(f"Corpus length: {len(ids):,} tokens")

# We want every position's activation for every neuron, but in CTX-sized
# windows. Strategy: walk over the corpus in non-overlapping CTX-sized
# windows, capture activations for every position in each window.
# Memory: per layer per window = CTX * D_MLP * 2 bytes (fp16) = 256*1024*2 = 512KB
#         Total across layers: 4 * 0.5 = 2MB per window.
#         Number of windows: 1.1M / 256 ≈ 4300. Total ≈ 8.6GB.
# Too much. Instead: sample windows, capture flatten, keep per-neuron top-K.

K = 8                      # top-K examples per neuron
N_WINDOWS = 800            # sample ~200K positions total
WIN = CTX
rng = np.random.default_rng(0)
starts = rng.integers(0, len(ids) - WIN, size=N_WINDOWS)

# Per-neuron top-K state: heap-like list of (-activation, layer, neuron_idx, abs_pos, snippet)
# Keep arrays per (layer, neuron) of length K, sorted descending.
# Use float16 to save memory. Shape (L, D_MLP, K).
top_act = -np.inf * np.ones((N_LAYERS, D_MLP, K), dtype=np.float32)
top_pos = np.zeros((N_LAYERS, D_MLP, K), dtype=np.int64)  # absolute char position in the corpus

t0 = time.time()
with torch.no_grad():
    for wi, st in enumerate(starts):
        st = int(st)
        window = ids[st:st+WIN].unsqueeze(0)  # 1, T
        _, acts = model(window)
        for L in range(N_LAYERS):
            a = acts[L][0]                     # T, D_MLP
            # For each neuron, find max over T and update top-K if needed.
            max_per_neuron, argmax_per_neuron = a.max(dim=0)
            mp = max_per_neuron.cpu().numpy().astype(np.float32)        # D_MLP
            ap = argmax_per_neuron.cpu().numpy().astype(np.int64)       # D_MLP
            # Compare each neuron's max to its current Kth-best
            for k in range(K):
                worst = top_act[L, :, k]
                better = mp > worst
                if better.any():
                    # For each "better" neuron, insert and shift down
                    idxs = np.where(better)[0]
                    for n in idxs:
                        new_act = mp[n]
                        new_pos = st + ap[n]
                        # find correct insert position
                        pos = np.searchsorted(-top_act[L, n], -new_act)
                        if pos < K:
                            top_act[L, n, pos+1:] = top_act[L, n, pos:K-1]
                            top_pos[L, n, pos+1:] = top_pos[L, n, pos:K-1]
                            top_act[L, n, pos] = new_act
                            top_pos[L, n, pos] = new_pos
                    break
        if wi % 100 == 0:
            print(f"  window {wi}/{N_WINDOWS}  elapsed {time.time()-t0:.1f}s")
print(f"Activation pass done in {time.time()-t0:.1f}s")

# ---------- pick "interesting" neurons ----------
# Heuristic: high peak activation + the K top activations are dispersed
# across the corpus (not clustered in one region — that suggests genuine
# content-feature, not a position artefact).
peak = top_act[:, :, 0]           # L, D_MLP
peak[~np.isfinite(peak)] = 0
disp = top_pos[:, :, :K].std(axis=2) / (len(ids) + 1)  # 0..1, higher = more spread out

# Score = peak * dispersion (rewards high firers that fire in many places)
score = peak * (0.3 + disp)
# Per layer, take top N
N_PER_LAYER = 8
chosen = []
for L in range(N_LAYERS):
    s = score[L]
    top_n = np.argsort(s)[::-1][:N_PER_LAYER]
    for n in top_n:
        if not np.isfinite(top_act[L, n, 0]) or top_act[L, n, 0] < 1.0:
            continue
        chosen.append((L, int(n)))
print(f"Chose {len(chosen)} neurons")

# ---------- gather context snippets with activations ----------
# For each chosen neuron, re-run the K windows around its top positions
# to get per-token activation values for highlighting.

CONTEXT_LEFT = 50  # chars before the activation position
CONTEXT_RIGHT = 10  # chars after

records = []
with torch.no_grad():
    for (L, n) in chosen:
        snippets = []
        for k in range(K):
            if not np.isfinite(top_act[L, n, k]) or top_act[L, n, k] < 0.5:
                break
            abs_pos = int(top_pos[L, n, k])
            # Build a window centered on abs_pos so the model has full context
            win_start = max(0, abs_pos - CTX + CONTEXT_RIGHT + 1)
            win_end = win_start + CTX
            if win_end > len(ids):
                win_end = len(ids)
                win_start = win_end - CTX
            window = ids[win_start:win_end].unsqueeze(0)
            _, acts = model(window)
            a = acts[L][0]                 # T, D_MLP
            an = a[:, n].cpu().numpy()     # T
            # Position of activation token within the window
            local_pos = abs_pos - win_start
            # Snippet to display: [CONTEXT_LEFT chars to left of activation .. activation .. CONTEXT_RIGHT chars right]
            start = max(0, local_pos - CONTEXT_LEFT)
            end   = min(CTX, local_pos + CONTEXT_RIGHT + 1)
            chars = [itos[int(t)] for t in window[0, start:end].cpu().numpy()]
            acts_s = an[start:end].tolist()
            snippets.append({
                "act": float(top_act[L, n, k]),
                "chars": chars,
                "act_per_char": acts_s,
                "highlight_pos": local_pos - start,
                "abs_pos": abs_pos,
            })
        if snippets:
            records.append({
                "layer": L,
                "neuron": int(n),
                "peak": float(top_act[L, n, 0]),
                "snippets": snippets,
            })

# Sort by layer then by peak descending
records.sort(key=lambda r: (r["layer"], -r["peak"]))

# Save
out_path = ANA / "neurons.json"
out_path.write_text(json.dumps(records, indent=2))
print(f"Saved {len(records)} neuron records to {out_path}")
print(f"Total time: {time.time()-t0:.1f}s")
