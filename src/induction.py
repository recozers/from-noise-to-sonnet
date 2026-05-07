"""
The induction-head experiment.

Setup (deliberately minimal, following Anthropic's "Induction Heads" paper):
  vocab        = 32 distinct tokens
  sequence     = [P random tokens] + [SAME P random tokens]  → length 2P = 64
  architecture = 2 layers, 2 heads each, d_model = 64       → 25 K params
  loss         = next-token prediction on the SECOND half only
                  (the first half is unpredictable; only the second half can
                   be solved, and only by induction)

We expect to see:
  - a "phase transition" in loss as the circuit assembles
  - a previous-token head emerge in L0
  - an induction head emerge in L1 (composing with L0's output)
"""
import os, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/home/stuart/sonnet_mind")
OUT  = ROOT / "induction_run"
OUT.mkdir(exist_ok=True)
(OUT / "ckpts").mkdir(exist_ok=True)

VOCAB    = 32
P        = 32                 # prefix length
SEQ      = 2 * P              # total seq length = 64
D_MODEL  = 64
N_HEADS  = 2
N_LAYERS = 2
HEAD_D   = D_MODEL // N_HEADS
BATCH    = 256
LR_MAX   = 1e-3
WD       = 0.0
ITERS    = 4000
WARMUP   = 100
LR_MIN   = 1e-4
DEVICE   = "cuda"
SEED     = 42

torch.manual_seed(SEED); np.random.seed(SEED)

def make_batch(B):
    # Random prefix of length P, repeated.
    pref = torch.randint(0, VOCAB, (B, P), device=DEVICE)
    seq = torch.cat([pref, pref], dim=1)  # B, 2P
    x = seq[:, :-1]
    y = seq[:, 1:]
    # Mask: only score loss on positions in the SECOND half.
    # Position t in the input corresponds to predicting token at t+1.
    # Second-half target positions are [P, 2P-1] in y (which is shifted).
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, P-1:] = True   # y[:, P-1] is the (P)th true token = first of repeat
    return x, y, mask

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv  = nn.Linear(D_MODEL, 3*D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(SEQ, SEQ)).view(1,1,SEQ,SEQ))
    def forward(self, x, return_attn=False):
        B,T,C = x.size()
        q,k,v = self.qkv(x).split(C, dim=2)
        q = q.view(B,T,N_HEADS,HEAD_D).transpose(1,2)
        k = k.view(B,T,N_HEADS,HEAD_D).transpose(1,2)
        v = v.view(B,T,N_HEADS,HEAD_D).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(HEAD_D)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        y = self.proj(y)
        return (y, att) if return_attn else y

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL); self.attn = Attn()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4*D_MODEL), nn.GELU(),
            nn.Linear(4*D_MODEL, D_MODEL))
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
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(SEQ, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, x, return_attn=False):
        B,T = x.size()
        pos = torch.arange(0,T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        attns=[]
        for b in self.blocks:
            if return_attn: h, a = b(h, return_attn=True); attns.append(a)
            else: h = b(h)
        h = self.lnf(h); logits = self.head(h)
        return (logits, attns) if return_attn else logits

model = TinyGPT().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {n_params/1e3:.1f}K | vocab={VOCAB} | seq={SEQ} | "
      f"prefix={P} | d={D_MODEL} | heads={N_HEADS} | layers={N_LAYERS}")

opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9,0.95), weight_decay=WD)
def lr_at(it):
    if it < WARMUP: return LR_MAX * it / WARMUP
    p = (it - WARMUP) / max(1, ITERS - WARMUP)
    return LR_MIN + 0.5*(LR_MAX-LR_MIN)*(1+math.cos(math.pi*p))

# Frequent log-spaced + linear early checkpoints to catch the phase transition
ckpt_steps = sorted(set(
    [0,1,2,3,4,5,7,10,15,20,30,40,50,60,70,80,90,100,
     120,140,160,180,200,250,300,350,400,500,600,700,800,
     1000,1200,1500,2000,2500,3000,3500,4000]
))
ckpt_steps = [s for s in ckpt_steps if s <= ITERS]
print(f"Will save {len(ckpt_steps)} checkpoints")

@torch.no_grad()
def eval_metrics():
    """Return: val loss on second half, induction score per (L,H)."""
    model.eval()
    # Generate fresh batch of test sequences
    x, y, mask = make_batch(64)
    logits, atts = model(x, return_attn=True)
    # CE on masked positions only
    flat_logits = logits.view(-1, VOCAB)
    flat_y = y.reshape(-1)
    flat_mask = mask.reshape(-1)
    loss_all = F.cross_entropy(flat_logits[flat_mask], flat_y[flat_mask])
    val_loss = float(loss_all)

    # Induction score: at positions p in [P, 2P-2] of the INPUT (which predict
    # tokens y[p] = pref[p-P+1]), the induction head should attend FROM p TO p-P+1.
    # NB: input position p corresponds to token x[p] = pref[p-P]; we want to look
    # back to position p-P+1 (the token that came right after the matching one
    # in the first copy).
    ind_scores = np.zeros((N_LAYERS, N_HEADS), dtype=np.float64)
    for L in range(N_LAYERS):
        a = atts[L]  # B, H, T, T
        for H in range(N_HEADS):
            head = a[:, H]  # B, T, T
            scores = []
            # input length T = SEQ - 1 = 63 (we predict next, so input is x[:,:-1])
            # Actually we used x = seq[:,:-1] so T = 2P-1 = 63
            # second-half input positions: [P-1, 2P-2] = [31, 62]
            # The "induction lookup target" for input pos p is the position whose
            # CONTENT equals what should come next. If we are at pos p in 2nd half,
            # the matching pos in 1st half is p-P, and the target to copy from
            # is p-P+1 (the next token in the first copy).
            for p in range(P, 2*P-1):  # input positions in the strict second half
                target = p - P + 1
                scores.append(head[:, p, target].mean().item())
            ind_scores[L,H] = float(np.mean(scores))
    model.train()
    return val_loss, ind_scores, atts

# Save attention on a fixed canonical batch every checkpoint, for animation
torch.manual_seed(0)  # fixed eval batch
fixed_x, fixed_y, fixed_mask = make_batch(1)
torch.manual_seed(SEED + 1)

@torch.no_grad()
def fixed_attn():
    model.eval()
    _, atts = model(fixed_x, return_attn=True)
    arr = np.stack([a[0].cpu().float().numpy() for a in atts])
    model.train()
    return arr

# ---------- training ----------
log = []
ckpt_set = set(ckpt_steps)
t0 = time.time()
model.train()
running = 0.0

torch.manual_seed(SEED)
for it in range(ITERS+1):
    if it in ckpt_set:
        vl, ind, _ = eval_metrics()
        atn = fixed_attn()
        np.save(OUT / "ckpts" / f"attn_{it:06d}.npy", atn.astype(np.float16))
        torch.save({"model": model.state_dict(), "step": it, "val_loss": vl,
                    "ind_scores": ind.tolist()},
                   OUT / "ckpts" / f"ckpt_{it:06d}.pt")
        elapsed = time.time() - t0
        # find max ind across heads
        maxind = float(ind.max())
        log.append({"step": it, "val_loss": vl, "ind_max": maxind,
                    "ind_per_head": ind.tolist(), "elapsed": elapsed})
        with (OUT / "log.json").open("w") as f: json.dump(log, f, indent=2)
        print(f"step {it:6d} | val_loss {vl:.4f} | max_ind {maxind:.3f} | "
              f"L0 {ind[0,0]:.3f},{ind[0,1]:.3f} | L1 {ind[1,0]:.3f},{ind[1,1]:.3f} | "
              f"elapsed {elapsed:6.1f}s")

    if it == ITERS: break

    lr = lr_at(it)
    for g in opt.param_groups: g["lr"] = lr

    x, y, mask = make_batch(BATCH)
    logits = model(x)
    flat_logits = logits.view(-1, VOCAB)
    flat_y = y.reshape(-1)
    flat_mask = mask.reshape(-1)
    loss = F.cross_entropy(flat_logits[flat_mask], flat_y[flat_mask])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

# Save metadata
(OUT / "meta.json").write_text(json.dumps({
    "vocab": VOCAB, "P": P, "seq": SEQ, "d_model": D_MODEL,
    "n_heads": N_HEADS, "n_layers": N_LAYERS, "iters": ITERS,
    "n_params": n_params,
}, indent=2))

print(f"\nDONE in {time.time()-t0:.1f}s")
print(f"Final val_loss: {log[-1]['val_loss']:.4f}")
print(f"Final max induction: {log[-1]['ind_max']:.3f}  "
      f"(perfect would be ≈ 1.0)")
