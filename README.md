# From Noise to Sonnet

A creative interpretability project on a remote RTX 3090. I was given a
six-hour compute budget; I used about fifty minutes of it (~8 of those on
the GPU itself). I trained a tiny GPT (3.24M parameters, 4 layers × 4
heads, char-level) on Shakespeare from random initialization, performed
mechanistic interpretability on the result, then trained a *second*
even-tinier model on a synthetic task that elicited a clean two-layer
induction circuit — the substrate of in-context learning, captured
forming in real time.

## Open the report first

**[`index.html`](index.html)** — the illustrated lab notebook. A
self-contained HTML file (~5.7 MB) with all figures embedded inline.
Open it in a browser. That is the deliverable.

If this repo is deployed (Vercel / GitHub Pages), the URL above serves
the report directly.

## What's here

```
sonnet_mind/
├── README.md               you are here
├── index.html       the illustrated lab notebook (5.7 MB, self-contained)
├── figs/                   14 figures total (PNGs + 2 animated GIFs)
│   ├── ...                 the Shakespeare experiment (loss, attention, embeddings)
│   ├── induction_phase.png ★ the phase transition
│   ├── induction_compose.png the canonical two-layer circuit
│   ├── induction_attn_grid.png attention through the transition
│   └── induction_anim.gif  the circuit assembling, animated
├── analysis/               raw interpretability artifacts (.npy, .json)
├── src/                    all the code I wrote
│   ├── train.py            Shakespeare model
│   ├── analyze.py, viz.py, extras.py
│   ├── induction.py        the synthetic induction-head experiment
│   ├── induction_viz.py    figures for it
│   └── make_report.py      compiles the HTML
├── meta.json               Shakespeare model config
├── log.json                val_loss vs step for every checkpoint
├── best_checkpoint.pt      best-val-loss weights (step 2424, val 1.570)
└── best_sample.txt         a sample generated from that checkpoint
```

## What the experiment actually did

**Phase 1 — train:** 10,000 steps of AdamW with cosine LR, no dropout,
batch size 64, context length 256. Took 7 minutes on the 3090. I saved
~40 log-spaced checkpoints, each one a full state dict + a 300-character
generation from a fixed Shakespeare probe + the 4×4×256×256 attention
tensor on that probe.

**Phase 2 — analyze:** loaded the best-val-loss checkpoint and computed:

- per-head behavioural fingerprints (previous-token / BOS / sharpness /
  entropy / induction)
- a doubled-random-sequence test for induction-head behaviour
- logit-lens analysis: applying the model's own unembed to the residual
  stream after each layer to see prediction sharpening
- character-embedding PCA (the model rediscovered the alphabet)
- position-embedding cosine-sim and 2D PCA (the model built itself a
  number line)

**Phase 3 — visualize:** matplotlib for everything. The animated attention
GIF is the centrepiece; the character-embedding PCA is the most surprising
result.

**Phase 4 — present:** `make_report.py` weaves the figures, samples, and
narrative into one HTML file with embedded base64 images.

## The single most interesting finding

The model, given only the constraint of predicting the next character,
spontaneously laid out character embeddings such that **uppercase letters
cluster together, lowercase letters cluster together, punctuation forms
its own region, and rare characters sit far from everything else**. No
labels, no dictionary, no rules — just cross-entropy bending the geometry
of `R^256` into a shape that knows which letters are kin. See
`figs/char_embed_pca.png`.

## The follow-up experiment: watching an induction head form

The Shakespeare model didn't grow a clean induction head — it didn't need
one. So I trained a second, even smaller model (105K params, 2 layers, 2
heads) on a synthetic copy task where induction is **the only winning
strategy**. The result is a textbook phase transition: loss flat for ~80
steps, then a 1.9-nat drop in ~150 steps as the L0 previous-token head
and the L1 induction head simultaneously crystallise into the canonical
two-layer composition circuit. See `figs/induction_phase.png` and
`figs/induction_compose.png`.

## Reproducing the run

On the GPU box (`ssh gpu`):

```bash
cd ~/sonnet_mind
source .venv/bin/activate
python train.py        # 7 min on RTX 3090
python analyze.py      # ~10 sec
python viz.py          # ~30 sec
python extras.py       # ~5 sec
python make_report.py  # <1 sec
```

All scripts assume the run lives at `~/sonnet_mind/run/`.
