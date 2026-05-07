"""
From Noise to Sonnet — train a tiny GPT on Shakespeare and capture its
learning trajectory: weights, generations, loss, attention on a fixed probe
at log-spaced steps.

Decoder-only transformer, character-level, ~3.5M params.
"""
import os, json, math, time, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- config ----------
ROOT = Path("/home/stuart/sonnet_mind")
DATA = ROOT / "shakespeare.txt"
OUT  = ROOT / "run"
OUT.mkdir(exist_ok=True)
(OUT / "ckpts").mkdir(exist_ok=True)

CTX        = 256          # context length
D_MODEL    = 256
N_HEADS    = 4
N_LAYERS   = 4
DROPOUT    = 0.0
BATCH      = 64
LR_MAX     = 3e-4
WD         = 0.1
ITERS      = 10_000
WARMUP     = 200
LR_MIN     = 3e-5
EVAL_EVERY = 250
DEVICE     = "cuda"
SEED       = 1337
PROBE      = "ROMEO:\nO, she doth teach the torches to burn bright"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- data ----------
text = DATA.read_text()
chars = sorted(list(set(text)))
vocab = len(chars)
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(t): return "".join(itos[int(i)] for i in t)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.95 * len(data))
train_data = data[:n]
val_data   = data[n:]

# Save tokenizer once
(OUT / "meta.json").write_text(json.dumps({
    "vocab": vocab, "ctx": CTX, "d_model": D_MODEL, "n_heads": N_HEADS,
    "n_layers": N_LAYERS, "iters": ITERS, "stoi": stoi, "itos": {str(k):v for k,v in itos.items()},
    "probe": PROBE, "seed": SEED,
}, indent=2))

def get_batch(split):
    src = train_data if split=="train" else val_data
    idx = torch.randint(0, len(src) - CTX - 1, (BATCH,))
    x = torch.stack([src[i:i+CTX] for i in idx]).to(DEVICE, non_blocking=True)
    y = torch.stack([src[i+1:i+1+CTX] for i in idx]).to(DEVICE, non_blocking=True)
    return x, y

# ---------- model ----------
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
        q = q.view(B,T,self.n_heads,self.head_dim).transpose(1,2)  # B,H,T,d
        k = k.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float("-inf"))
        att = F.softmax(att, dim=-1)
        attn_out = att
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.proj(y)
        if return_attn:
            return y, attn_out
        return y

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, 4*D_MODEL),
            nn.GELU(),
            nn.Linear(4*D_MODEL, D_MODEL),
        )
    def forward(self, x, return_attn=False):
        if return_attn:
            a, attn = self.attn(self.ln1(x), return_attn=True)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x, attn
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(CTX, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab, bias=False)
        # weight tying
        self.head.weight = self.tok.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_attn=False):
        B,T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        attns = []
        for b in self.blocks:
            if return_attn:
                x, a = b(x, return_attn=True)
                attns.append(a)
            else:
                x = b(x)
        x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_attn:
            return logits, loss, attns
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=200, temperature=0.8, top_k=None):
        for _ in range(max_new):
            idx_cond = idx[:, -CTX:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

model = TinyGPT().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {n_params/1e6:.2f}M | vocab={vocab} | ctx={CTX} | d={D_MODEL} | heads={N_HEADS} | layers={N_LAYERS}")

# ---------- optimizer ----------
opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=WD)
def lr_at(it):
    if it < WARMUP: return LR_MAX * it / WARMUP
    p = (it - WARMUP) / max(1, ITERS - WARMUP)
    return LR_MIN + 0.5*(LR_MAX-LR_MIN)*(1+math.cos(math.pi*p))

# ---------- log-spaced checkpoint schedule ----------
# Want: many checkpoints early, fewer later. Log-spaced from 0 to ITERS.
ckpt_steps = sorted(set([0] + [int(round(x)) for x in np.unique(np.geomspace(1, ITERS, num=40).astype(int))] + [ITERS]))
ckpt_steps = [s for s in ckpt_steps if 0 <= s <= ITERS]
print(f"Will save {len(ckpt_steps)} checkpoints at steps: {ckpt_steps[:5]}...{ckpt_steps[-3:]}")

# ---------- helper: snapshot ----------
@torch.no_grad()
def snapshot(it):
    model.eval()
    # generate from probe (deterministic-ish: top_k=None, temp=0.8)
    ids = torch.tensor([encode(PROBE)], dtype=torch.long, device=DEVICE)
    out = model.generate(ids, max_new=300, temperature=0.8, top_k=None)
    sample = decode(out[0].tolist())

    # capture attention on probe (forward only, full ctx)
    # ensure probe fits in ctx
    probe_ids = torch.tensor([encode(PROBE)[:CTX]], dtype=torch.long, device=DEVICE)
    _, _, attns = model(probe_ids, return_attn=True)
    # attns: list of [1, H, T, T]
    attn_arr = np.stack([a[0].detach().cpu().float().numpy() for a in attns])  # L,H,T,T

    # val loss
    losses = []
    for _ in range(8):
        xb, yb = get_batch("val")
        _, l = model(xb, yb)
        losses.append(l.item())
    vl = float(np.mean(losses))

    # save model weights (small)
    torch.save({"model": model.state_dict(), "step": it, "val_loss": vl}, OUT / "ckpts" / f"ckpt_{it:06d}.pt")
    np.save(OUT / "ckpts" / f"attn_{it:06d}.npy", attn_arr.astype(np.float16))
    (OUT / "ckpts" / f"sample_{it:06d}.txt").write_text(sample)
    model.train()
    return vl, sample

# ---------- training loop ----------
log = []
ckpt_set = set(ckpt_steps)
t0 = time.time()
model.train()
running = 0.0

for it in range(ITERS+1):
    # snapshot first
    if it in ckpt_set:
        vl, sample = snapshot(it)
        elapsed = time.time() - t0
        print(f"step {it:6d} | val_loss {vl:.4f} | elapsed {elapsed:6.1f}s | sample[:80]: {sample[len(PROBE):len(PROBE)+80].replace(chr(10),'/')}")
        log.append({"step": it, "val_loss": vl, "elapsed": elapsed})
        with (OUT / "log.json").open("w") as f: json.dump(log, f, indent=2)

    if it == ITERS: break

    lr = lr_at(it)
    for g in opt.param_groups: g["lr"] = lr

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    running = 0.99*running + 0.01*loss.item() if running else loss.item()

    if it % 100 == 0:
        elapsed = time.time() - t0
        print(f"  it {it:6d} | tr_loss {running:.4f} | lr {lr:.2e} | elapsed {elapsed:6.1f}s")

print(f"\nDONE. total time: {time.time()-t0:.1f}s")
