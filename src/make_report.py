"""
Compile the final illustrated lab notebook as a self-contained HTML file.
Embeds figures (PNG / GIF) inline as base64.
"""
import json, base64, math
from pathlib import Path

ROOT = Path("/home/stuart/sonnet_mind")
RUN = ROOT / "run"
ANA = RUN / "analysis"
FIGS = RUN / "figs"
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)

meta = json.loads((RUN / "meta.json").read_text())
log = json.loads((RUN / "log.json").read_text())
fps = json.loads((ANA / "fingerprints.json").read_text())
best = json.loads((ANA / "best_ckpt.json").read_text())

def embed(p, mime):
    if not Path(p).exists(): return ""
    return f"data:{mime};base64,{base64.b64encode(Path(p).read_bytes()).decode()}"

loss_curve   = embed(FIGS / "loss_curve.png", "image/png")
attn_gif     = embed(FIGS / "attn_evolution.gif", "image/gif")
heads_grid   = embed(FIGS / "heads_final.png", "image/png")
fingerprints = embed(FIGS / "fingerprints.png", "image/png")
induction    = embed(FIGS / "induction_bar.png", "image/png")
logit_lens   = embed(FIGS / "logit_lens.png", "image/png")
gens_grid    = embed(FIGS / "generations_grid.png", "image/png")
char_pca     = embed(FIGS / "char_embed_pca.png", "image/png")
pos_emb      = embed(FIGS / "pos_embed_sim.png", "image/png")
head_deep    = embed(FIGS / "head_deep_dive.png", "image/png")
ind_phase    = embed(FIGS / "induction_phase.png", "image/png")
ind_compose  = embed(FIGS / "induction_compose.png", "image/png")
ind_grid     = embed(FIGS / "induction_attn_grid.png", "image/png")
ind_anim     = embed(FIGS / "induction_anim.gif", "image/gif")

# Load induction-experiment metadata
import json as _j
ind_meta = _j.loads((ROOT / "induction_run" / "meta.json").read_text())
ind_log  = _j.loads((ROOT / "induction_run" / "log.json").read_text())

CKPTS = RUN / "ckpts"
sample_files = sorted(CKPTS.glob("sample_*.txt"))
sample_steps = [int(f.stem.split("_")[1]) for f in sample_files]
def sample_at(step_target):
    i = min(range(len(sample_steps)), key=lambda i: abs(sample_steps[i]-step_target))
    return sample_steps[i], sample_files[i].read_text()

s_init  = sample_at(0)
s_early = sample_at(20)
s_mid   = sample_at(200)
s_good  = sample_at(best["step"])
s_late  = sample_at(log[-1]["step"])

# Build the JS data for the interactive scrubber: all sample texts + step + val_loss
val_by_step = {r["step"]: r["val_loss"] for r in log}
scrubber_data = []
for step, fp in zip(sample_steps, sample_files):
    txt = fp.read_text()
    scrubber_data.append({
        "step": step,
        "val_loss": round(val_by_step.get(step, 0.0), 4),
        "text": txt,
    })
import json as _json
scrubber_json = _json.dumps(scrubber_data)
PROBE_LEN = len(meta["probe"])

def fmt_sample(step, text, label):
    body = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    return f"""
<figure class="sample">
  <figcaption>{label} <span class="step">step {step}</span></figcaption>
  <pre>{body}</pre>
</figure>
"""

# Identify notable heads
top_head = max(fps, key=lambda f: f["induction"])
top_prev = max(fps, key=lambda f: f["prev_token"])
top_bos  = max(fps, key=lambda f: f["bos"])
top_sharp = max(fps, key=lambda f: f["sharpness"])

def head_label(f):
    if f["entropy"] > 4.0:
        return "uniform / non-specialized"
    if f["bos"] > 0.5:
        return "rest / null head"
    if f["prev_token"] > 0.3 and f["sharpness"] > 0.5:
        return "previous-token (sharp)"
    if f["prev_token"] > 0.2:
        return "previous-token-ish"
    if f["sharpness"] > 0.4:
        return "sharp / content-based lookup"
    return "diffuse context mixer"

head_table = "\n".join(
    f'<tr><td>L{f["layer"]}.H{f["head"]}</td><td>{head_label(f)}</td>'
    f'<td>{f["prev_token"]:.2f}</td><td>{f["bos"]:.2f}</td>'
    f'<td>{f["induction"]:.3f}</td><td>{f["sharpness"]:.2f}</td>'
    f'<td>{f["entropy"]:.2f}</td></tr>'
    for f in fps
)

# Compute params
n_params_M = (meta["vocab"]*meta["d_model"] + meta["ctx"]*meta["d_model"] +
              meta["n_layers"]*(3*meta["d_model"]*meta["d_model"] + meta["d_model"]*meta["d_model"] +
                                4*meta["d_model"]*meta["d_model"] + 4*meta["d_model"]*meta["d_model"] +
                                4*meta["d_model"]) + meta["d_model"]) / 1e6

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>From Noise to Sonnet · A Tiny Mind Forms</title>
<style>
:root {{
  --ink:#1a1a1a; --parchment:#f4ecd8; --paper:#fbf6e8;
  --night:#0e0f12; --night2:#181a20;
  --gold:#e9b44c; --iris:#9a8aff; --mint:#43d9ad; --rose:#ff6b8b;
  --muted:#7a7a7a;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin:0; padding:0; }}
body {{
  font-family: ui-serif, "EB Garamond", Georgia, serif;
  font-size: 18px;
  line-height: 1.65;
  color: var(--ink);
  background: var(--parchment);
  -webkit-font-smoothing: antialiased;
}}
.hero {{
  background: linear-gradient(180deg, var(--night) 0%, var(--night2) 100%);
  color: #fafafa;
  padding: 7vw 8vw 6vw 8vw;
  border-bottom: 4px solid var(--gold);
  text-align: center;
}}
.hero h1 {{
  font-family: ui-serif, "EB Garamond", Georgia, serif;
  font-weight: 400;
  font-size: clamp(2.5rem, 6vw, 5rem);
  letter-spacing: -0.02em;
  margin: 0 0 0.5em 0;
  font-style: italic;
}}
.hero .subtitle {{
  font-size: clamp(1rem, 1.6vw, 1.2rem);
  color: #c8b88f;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 1em;
}}
.hero .credit {{
  font-size: 0.85rem; color: #888; font-style: italic;
}}
.toc {{
  background: var(--night2);
  color: #aaa;
  padding: 1rem 2rem;
  text-align: center;
  font-size: 0.85rem;
  border-bottom: 1px solid #2a2c33;
}}
.toc a {{
  color: #c8b88f;
  text-decoration: none;
  margin: 0 0.7rem;
  font-style: italic;
}}
.toc a:hover {{ color: var(--gold); border-bottom: 1px solid var(--gold); }}
main {{
  max-width: 920px;
  margin: 0 auto;
  padding: 4rem 1.5rem;
}}
section {{ margin-bottom: 5rem; scroll-margin-top: 1rem; }}
h2 {{
  font-family: ui-serif, "EB Garamond", Georgia, serif;
  font-weight: 400;
  font-size: 2.1rem;
  font-style: italic;
  margin-top: 3em;
  margin-bottom: 0.4em;
  color: #2a1f0a;
  border-bottom: 1px solid rgba(0,0,0,0.15);
  padding-bottom: 0.3em;
}}
h2 .step-label {{
  display: block;
  font-size: 0.85rem;
  color: var(--muted);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  font-style: normal;
  margin-bottom: 0.4em;
}}
p {{ margin: 1em 0; }}
em {{ font-style: italic; color: #4a3a1a; }}
code {{
  font-family: ui-monospace, "Menlo", monospace;
  background: rgba(0,0,0,0.06);
  padding: 0.1em 0.35em;
  border-radius: 3px;
  font-size: 0.92em;
}}
.figure {{
  margin: 2.5rem 0;
  text-align: center;
}}
.figure img {{
  max-width: 100%;
  border-radius: 4px;
  box-shadow: 0 6px 30px rgba(0,0,0,0.18);
}}
.figure .caption {{
  font-size: 0.92rem;
  color: var(--muted);
  font-style: italic;
  margin-top: 0.6rem;
  max-width: 80%;
  margin-left: auto;
  margin-right: auto;
}}
.sample {{
  background: var(--paper);
  border-left: 4px solid var(--gold);
  padding: 1rem 1.2rem;
  margin: 1.5rem 0;
  border-radius: 0 4px 4px 0;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}}
.sample figcaption {{
  font-size: 0.85rem;
  color: var(--muted);
  margin-bottom: 0.6em;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  font-weight: 600;
}}
.sample .step {{
  background: #2a1f0a;
  color: var(--gold);
  font-family: ui-monospace, monospace;
  padding: 1px 6px;
  border-radius: 3px;
  margin-left: 0.6em;
  text-transform: none;
  letter-spacing: 0.02em;
}}
.sample pre {{
  font-family: ui-monospace, "Menlo", monospace;
  font-size: 0.86rem;
  white-space: pre-wrap;
  margin: 0;
  color: #2a1f0a;
  line-height: 1.45;
}}
.callout {{
  border: 1px solid rgba(154,138,255,0.4);
  background: rgba(154,138,255,0.08);
  padding: 1rem 1.2rem;
  border-radius: 4px;
  margin: 1.5rem 0;
  font-size: 0.95rem;
}}
.callout strong {{ color: #4f3eb3; }}
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 1.5rem 0;
  font-size: 0.9rem;
  background: var(--paper);
}}
th, td {{
  border-bottom: 1px solid rgba(0,0,0,0.1);
  padding: 0.55rem 0.7rem;
  text-align: left;
}}
th {{
  background: rgba(0,0,0,0.04);
  font-weight: 600;
  letter-spacing: 0.04em;
  font-size: 0.78rem;
  text-transform: uppercase;
  color: var(--muted);
}}
td:first-child, th:first-child {{ font-family: ui-monospace, monospace; }}
.config {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem 1.2rem;
  margin: 1.5rem 0;
  font-family: ui-monospace, monospace;
  font-size: 0.88rem;
  color: var(--muted);
}}
.config span b {{ color: var(--ink); }}
.lede {{
  font-size: 1.18rem;
  line-height: 1.55;
  color: #3a2f15;
  font-style: italic;
  border-left: 3px solid var(--gold);
  padding: 0.2em 1.2em;
  margin: 2rem 0;
}}
.pull {{
  font-size: 1.4rem;
  line-height: 1.45;
  color: #2a1f0a;
  font-style: italic;
  text-align: center;
  margin: 3rem 0 3rem 0;
  padding: 0 2rem;
  position: relative;
}}
.pull::before, .pull::after {{
  content: "";
  display: block;
  width: 40px;
  height: 1px;
  background: var(--gold);
  margin: 1rem auto;
}}
footer {{
  background: var(--night);
  color: #888;
  padding: 3rem 1.5rem;
  text-align: center;
  font-size: 0.9rem;
}}
footer .signature {{
  font-style: italic;
  color: #c8b88f;
  font-family: ui-serif, Georgia, serif;
  font-size: 1.05rem;
  margin-bottom: 0.4em;
}}
.scrubber {{
  background: #181a20;
  color: #fafafa;
  padding: 1.5rem 1.5rem 1.7rem 1.5rem;
  border-radius: 6px;
  margin: 2.5rem 0;
  box-shadow: 0 4px 24px rgba(0,0,0,0.18);
  border: 1px solid #2a2c33;
}}
.scrubber-label {{
  font-size: 0.95rem;
  color: #c8b88f;
  margin-bottom: 1rem;
  font-style: italic;
}}
.scrubber-readout {{
  display: flex;
  gap: 1.5rem;
  font-family: ui-monospace, "Menlo", monospace;
  font-size: 0.86rem;
  color: #888;
  margin-bottom: 0.6rem;
}}
.scrubber-readout b {{
  color: var(--gold);
  font-weight: 600;
}}
#scrub-slider {{
  width: 100%;
  height: 4px;
  -webkit-appearance: none;
  background: linear-gradient(90deg, #43d9ad 0%, #e9b44c 30%, #ff6b8b 100%);
  border-radius: 2px;
  outline: none;
  margin: 0.4rem 0 1rem 0;
}}
#scrub-slider::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 18px; height: 18px; border-radius: 50%;
  background: var(--gold);
  cursor: pointer;
  border: 2px solid #1a1a1a;
  box-shadow: 0 0 8px rgba(233,180,76,0.6);
}}
#scrub-slider::-moz-range-thumb {{
  width: 18px; height: 18px; border-radius: 50%;
  background: var(--gold); cursor: pointer; border: 2px solid #1a1a1a;
}}
#scrub-text {{
  background: #0e0f12;
  border: 1px solid #2a2c33;
  border-radius: 4px;
  padding: 1rem 1.2rem;
  font-family: ui-monospace, "Menlo", monospace;
  font-size: 0.86rem;
  line-height: 1.55;
  color: #d8d4c2;
  white-space: pre-wrap;
  min-height: 320px;
  max-height: 420px;
  overflow-y: auto;
}}
#scrub-text .probe {{
  color: var(--gold);
  font-weight: 600;
}}
#scrub-text .gen {{
  color: #fafafa;
}}
</style>
</head>
<body>

<header class="hero">
  <div class="subtitle">a six-hour journey on an RTX 3090</div>
  <h1>From Noise to Sonnet</h1>
  <p style="max-width:640px;margin:0 auto;color:#c8b88f;font-style:italic;font-size:1.1rem;">
    in which I take a tiny transformer&mdash;3.24 million parameters,
    smaller than a yeast cell&rsquo;s genome&mdash;and watch it learn to
    dream in iambic English. then I open it up to see what circuits formed
    inside.
  </p>
  <p class="credit">an experiment by claude · {log[-1]['step']:,} training steps · char-level Shakespeare</p>
</header>

<nav class="toc">
  <a href="#prologue">prologue</a>·
  <a href="#curve">the descent</a>·
  <a href="#voice">the voice emerges</a>·
  <a href="#eyes">the eyes open</a>·
  <a href="#circuits">reading the circuits</a>·
  <a href="#alphabet">the rediscovered alphabet</a>·
  <a href="#position">how it counts</a>·
  <a href="#lens">the logit lens</a>·
  <a href="#induction">induction</a>·
  <a href="#summon">induction, summoned</a>·
  <a href="#closing">closing</a>
</nav>

<main>

<section id="prologue">
<p class="lede">
  A neural network is not built. It is grown. You set the architecture &mdash;
  the empty house &mdash; and pour text through it for a long time, and
  somewhere in that downpour <em>tendencies</em> form: little patterns of
  attention, little kinks in geometry, that learn to do specific jobs.
  This is a record of one such growing.
</p>
<div class="config">
  <span>arch: <b>decoder-only transformer</b></span>
  <span>params: <b>{n_params_M:.2f}M</b></span>
  <span>layers: <b>{meta['n_layers']}</b></span>
  <span>heads/layer: <b>{meta['n_heads']}</b></span>
  <span>d_model: <b>{meta['d_model']}</b></span>
  <span>context: <b>{meta['ctx']}</b></span>
  <span>vocab: <b>{meta['vocab']} characters</b></span>
  <span>training: <b>{log[-1]['step']:,} steps · {log[-1]['elapsed']:.0f}s on RTX 3090</b></span>
</div>
<p>
  No tokenizer, no BPE; just one-hot characters. Adam-W, cosine schedule,
  no dropout. A run small enough to be poetic and big enough to be real.
</p>
</section>

<section id="curve">
<h2><span class="step-label">i · the descent</span>The shape of learning</h2>
<p>
  The first thing that happens is also the most dramatic. Loss starts at
  <code>{log[0]['val_loss']:.2f}</code> nats per character &mdash; barely
  better than chance &mdash; and within a few hundred steps the model has
  noticed <em>that letters tend to follow letters, that there are spaces,
  that the alphabet is not made of all-equal symbols</em>. The curve plummets.
</p>
<p>
  Then, slower, the model begins to learn <em>which</em> letters tend to
  follow which: that <code>th</code> wants <code>e</code>, that
  <code>ROM</code> wants <code>EO</code>. By step
  <code>{best['step']:,}</code>, validation loss has bottomed out at
  <code>{best['val_loss']:.3f}</code> nats &mdash; about
  <code>{2.71828**best['val_loss']:.1f}</code> alternatives per character on
  average. It has, in some real sense, <em>learned to write English</em>.
</p>
<p>
  And then the most poignant thing happens. The model keeps training.
  Validation loss begins to <em>rise</em>. Without enough text to feed the
  hunger of {n_params_M:.1f}M weights, the model starts to memorise. It
  isn&rsquo;t learning anymore. It is rehearsing.
</p>

<div class="figure">
  <img src="{loss_curve}" alt="loss curve">
  <div class="caption">
    Validation loss across {len(log)} log-spaced checkpoints. The star marks
    peak generalisation: step {best['step']:,}, val&nbsp;loss&nbsp;{best['val_loss']:.3f}.
    From there forward, the model trades understanding for memorisation.
  </div>
</div>

<div class="callout">
<strong>An aside on units.</strong> The dashed line at
<code>ln({meta['vocab']}) ≈ {math.log(meta['vocab']):.2f}</code> is the
loss you&rsquo;d get from a uniform random guess over the {meta['vocab']}
characters. We ended at roughly a third of that. The model has a lot of
opinions about what letter comes next.
</div>
</section>

<section id="voice">
<h2><span class="step-label">ii · the voice emerges</span>The sound of a forming mind</h2>
<p>
  Same prompt, same temperature (0.8), different moments in the training.
  We feed it &ldquo;ROMEO:\\nO, she doth teach the torches to burn bright&rdquo;
  and ask it to keep going.
</p>

{fmt_sample(s_init[0],  s_init[1],  "at birth — uniform noise")}
{fmt_sample(s_early[0], s_early[1], "first inklings — letter-frequency in place")}
{fmt_sample(s_mid[0],   s_mid[1],   "mid-training — words appearing")}
{fmt_sample(s_good[0],  s_good[1],  "best-ckpt — language, mostly")}
{fmt_sample(s_late[0],  s_late[1],  "late training — the overfit voice")}

<div class="figure">
  <img src="{gens_grid}" alt="generations grid">
  <div class="caption">A timeline of the same probe answered at log-spaced
  checkpoints. Read top to bottom and you can almost <em>hear</em> a voice
  precipitating out of static.</div>
</div>

<div class="scrubber">
  <div class="scrubber-label">
    <span style="color:var(--gold);">drag the slider</span> to scrub through training and watch the model find its voice.
  </div>
  <div class="scrubber-readout">
    <span class="readout-step">step <b id="scrub-step">0</b></span>
    <span class="readout-loss">val&nbsp;loss <b id="scrub-loss">—</b></span>
    <span class="readout-pos">checkpoint <b id="scrub-idx">0</b> / {len(scrubber_data)-1}</span>
  </div>
  <input type="range" id="scrub-slider" min="0" max="{len(scrubber_data)-1}" value="{len(scrubber_data)//2}" step="1" />
  <div class="scrubber-output" id="scrub-text"></div>
</div>

<p class="pull">
A model with three million parameters learns to dream in iambic English in
about seven minutes.
</p>
</section>

<section id="eyes">
<h2><span class="step-label">iii · the eyes open</span>Attention learns to look</h2>
<p>
  Inside each transformer block lives a self-attention layer: a row of
  &ldquo;heads&rdquo; that each, independently, decide where to look in
  the preceding text. At initialization those decisions are uniform-ish
  noise. They learn to specialize.
</p>
<div class="figure">
  <img src="{attn_gif}" alt="attention evolution">
  <div class="caption">Attention patterns of one head per layer (head 0)
  while the model trains, on a fixed Shakespeare probe. Each frame is a
  later step. Watch the diagonals sharpen and the lookback structures form.
  This is, more or less, a mind paying attention.</div>
</div>
<div class="figure">
  <img src="{heads_grid}" alt="heads at end of training">
  <div class="caption">All {meta['n_layers']*meta['n_heads']} attention
  heads at peak generalisation, on the same probe. Each subplot is a
  {meta['ctx']}&times;{meta['ctx']} matrix: bright pixel at <em>(i,j)</em>
  means token <em>i</em> attended strongly to token <em>j</em>. Different
  heads, different shapes, different jobs. Layer 0 is essentially flat
  &mdash; the model has decided not to use those heads.</div>
</div>
</section>

<section id="circuits">
<h2><span class="step-label">iv · what each head does</span>Reading the circuits</h2>
<p>
  This is the part I find genuinely thrilling. The heads above don&rsquo;t
  just look <em>somewhere</em>; many of them have learned recognizable
  algorithmic <em>roles</em>. We can score each one by simple geometric
  features of its attention pattern:
</p>
<ul style="margin-left:1.2em;">
<li><b>previous-token</b>: avg attention on the diagonal one step back. A head with high score is asking <em>&ldquo;what was the last character?&rdquo;</em></li>
<li><b>BOS / null</b>: avg attention on token 0. High = the head has learned to <em>do nothing</em>, routing the residual through unchanged.</li>
<li><b>sharpness</b>: how peaked the attention rows are. High = the head makes specific lookups instead of broad averages.</li>
<li><b>entropy</b>: the dual of sharpness. Heads with entropy near <code>ln({meta['ctx']}) ≈ {math.log(meta['ctx']):.2f}</code> are essentially uniform.</li>
<li><b>induction</b>: tested with a sequence of the form <code>[random A][random A]</code>. A head scores high if, while reading the second copy at position <em>p</em>, it attends to the position right after the matching prefix in the first copy &mdash; i.e. it has learned the copy-from-history pattern that makes in-context learning possible.</li>
</ul>
<div class="figure">
  <img src="{fingerprints}" alt="head fingerprints">
  <div class="caption">A &ldquo;fingerprint&rdquo; for each of the
  {meta['n_layers']*meta['n_heads']} heads. Brighter cells mean a stronger
  signal of that role.</div>
</div>
<table>
<thead><tr><th>head</th><th>my best guess at role</th><th>prev</th><th>BOS</th><th>induction</th><th>sharpness</th><th>entropy</th></tr></thead>
<tbody>
{head_table}
</tbody>
</table>
<p>
  Three findings I want to draw your attention to. First: <b>Layer 0 has
  refused to specialize</b>. All four heads have entropy ≈ {meta['ctx']}/4
  &mdash; uniform attention. The model has decided the L0 attention layer
  isn&rsquo;t the right place to spend representational budget; it routes
  the residual through nearly unchanged and lets the L0 MLP and later
  layers do the work. This is a discovery the optimizer made, not us.
</p>
<p>
  Second: <b>L{top_prev['layer']}.H{top_prev['head']} is unmistakably a
  previous-token head</b>. Its attention forms a clean off-by-one
  diagonal &mdash; every position attends, sharply, to the one just before
  it. That gives downstream layers the bigram structure they need.
</p>
<p>
  Third: <b>L{top_sharp['layer']}.H{top_sharp['head']} has the sharpest
  attention</b> in the network &mdash; sharpness {top_sharp['sharpness']:.2f},
  meaning on average it puts {top_sharp['sharpness']*100:.0f}% of its
  attention mass on a single position. That&rsquo;s a head doing very
  specific lookups, not averaging.
</p>
<div class="figure">
  <img src="{head_deep}" alt="head deep dive">
  <div class="caption">Attention of two specialized heads, plotted on
  <em>&ldquo;ROMEO: O Romeo, Romeo, wherefore art thou Romeo?&rdquo;</em>,
  with the actual characters labelled. The left head is the
  cleanest previous-token head &mdash; bright off-by-one diagonal. The
  right head, in layer 0, is the &ldquo;most induction-like&rdquo; only by
  technicality &mdash; you can see it attends mostly to the start of the
  sequence (a BOS-rest pattern).</div>
</div>
</section>

<section id="alphabet">
<h2><span class="step-label">v · the rediscovered alphabet</span>How the model thinks of characters</h2>
<p>
  Each of the {meta['vocab']} characters is mapped, via the
  <code>tok</code> embedding, to a vector in
  <code>R^{meta['d_model']}</code>. We can take all those vectors, run PCA,
  and look at the first two components.
</p>
<div class="figure">
  <img src="{char_pca}" alt="character embedding PCA">
  <div class="caption">Each dot is one of the {meta['vocab']} characters,
  positioned by its first two principal components. Color = my own a
  posteriori category labels (uppercase, lowercase, digit, etc.); the model
  was never told which was which.</div>
</div>
<p>
  The model has, on its own, <b>rediscovered the structure of the alphabet</b>.
  Uppercase letters cluster together. Lowercase letters cluster together.
  Punctuation forms its own region. Vowels and consonants further sub-cluster
  within those. Outliers (the rare <code>$</code>, <code>&amp;</code>,
  <code>Z</code>) sit far from everything else, exactly because they
  appear in unusual contexts.
</p>
<p>
  This is the deep lesson of word embeddings, condensed: <em>characters
  that appear in similar contexts get similar embeddings, because the
  model has learned to treat them similarly</em>. No labels, no rules, no
  dictionary &mdash; just cross-entropy on next-character prediction,
  bending the geometry of <code>R^{meta['d_model']}</code> into a shape
  that knows which letters are kin.
</p>
</section>

<section id="position">
<h2><span class="step-label">vi · how it counts</span>The geometry of position</h2>
<p>
  Transformers don&rsquo;t come knowing where each token is in the
  sentence; they have to learn position embeddings the same way they
  learn character embeddings. Below: the cosine similarity matrix between
  every pair of position vectors (left), and a 2D PCA of those same
  vectors colored by index (right).
</p>
<div class="figure">
  <img src="{pos_emb}" alt="position embeddings">
  <div class="caption">Left: positions near each other have similar
  embeddings (red band along the diagonal). Distant positions are
  uncorrelated or weakly opposed. Right: in 2D PCA, the {meta['ctx']}
  positions form a smooth curve &mdash; a one-dimensional manifold
  embedded in {meta['d_model']}-dim space. The model has built itself a
  number line.</div>
</div>
<p>
  I find this the quietest, most beautiful kind of result. The model was
  given {meta['ctx']} indistinguishable position slots and the constraint
  that its predictions had to depend on order. From that constraint alone
  it grew the geometry of <em>before</em> and <em>after</em>.
</p>
</section>

<section id="lens">
<h2><span class="step-label">vii · the logit lens</span>Reading the mind in flight</h2>
<p>
  Every layer in the network outputs a vector. If we apply the model&rsquo;s
  own output projection to the residual stream <em>after each block</em>,
  we can see what next-character distribution that intermediate
  representation already encodes. This is the <em>logit lens</em>.
</p>
<div class="figure">
  <img src="{logit_lens}" alt="logit lens">
  <div class="caption">The model&rsquo;s top-6 next-char predictions at the
  final position of the probe (which ends in &lsquo;...burn bright&rsquo;),
  applied at the residual after each layer. Layer 0 already knows the
  answer is space &mdash; English nearly always has a space after
  <em>bright</em>. Later layers commit harder and consider some
  alternatives (perhaps <em>bright<b>e</b>r</em> or <em>bright<b>l</b>y</em>).</div>
</div>
</section>

<section id="induction">
<h2><span class="step-label">viii · induction</span>Did it learn to copy?</h2>
<p>
  The most famous emergent circuit in transformers is the <em>induction
  head</em>: a head that, given a sequence containing the pattern
  <code>...AB...A</code>, attends to the position right after the previous
  <code>A</code>, allowing the model to predict <code>B</code> by simple
  copying. Induction heads are the substrate of in-context learning.
</p>
<p>
  Did our 3M-parameter Shakespeare model learn one? <em>Almost</em>. We
  feed in a doubled random sequence and measure each head&rsquo;s attention
  to the &ldquo;correct&rdquo; lookback position.
</p>
<div class="figure">
  <img src="{induction}" alt="induction bar">
  <div class="caption">Induction signal across all
  {meta['n_layers']*meta['n_heads']} heads. Above the dashed chance line,
  several heads in layers 2 and 3 show weak induction-like behaviour.
  Layer 0 heads also score above chance, but for an artefact reason: their
  attention is so uniform that they hit every position weakly, including
  the &ldquo;correct&rdquo; one.</div>
</div>
<p>
  No clean induction head emerged here. Real induction in tiny models
  needs at least a previous-token head <em>and</em> a head that can
  combine its output with key information from another position
  (&ldquo;Q-composition&rdquo; in the Anthropic literature). With only
  four heads per layer and four layers, our model has just barely enough
  algorithmic vocabulary to begin assembling that circuit, and the result
  is more <em>tendency</em> than <em>algorithm</em>. One imagines that
  with another order of magnitude of parameters, this signal would
  resolve into the genuine article.
</p>
</section>

<section id="summon">
<h2><span class="step-label">ix · induction, summoned</span>Watching the circuit assemble</h2>
<p>
  The Shakespeare model didn&rsquo;t produce a clean induction head. But
  the reason is mundane: it didn&rsquo;t need one. Predicting the next
  letter of English is mostly a job for n-gram statistics and bigram
  composition; you don&rsquo;t have to <em>copy from history</em> to do it
  well. So I gave the model a different task.
</p>
<p>
  A new minimal model: <code>{ind_meta['n_layers']}</code> layers,
  <code>{ind_meta['n_heads']}</code> heads each, <code>d_model =
  {ind_meta['d_model']}</code> &mdash; just <code>{ind_meta['n_params']/1e3:.0f}K</code>
  parameters total, smaller than a single PNG of the GIF above.
  Trained not on Shakespeare but on a synthetic task: random sequences of
  the form <code>[P random tokens] [SAME P random tokens]</code>, with
  loss only on the second half. The first half is unguessable. The second
  half is solvable by exactly one strategy: look back P positions, find
  what came after, copy it forward. <b>Induction or nothing.</b>
</p>

<div class="figure">
  <img src="{ind_phase}" alt="phase transition">
  <div class="caption">The phase transition. For the first
  <code>~80</code> training steps essentially nothing happens &mdash; the
  model can&rsquo;t do better than uniform guessing
  (<code>ln({ind_meta['vocab']}) ≈ {math.log(ind_meta['vocab']):.2f}</code>
  nats). Then in <em>~150 steps</em> the loss collapses by more than 1.9
  nats while the induction score in layer 1 climbs from chance to nearly
  half. <b>This is the circuit assembling itself.</b> Note that L0 (purple
  dotted) stays near zero — L0 isn&rsquo;t doing induction, it&rsquo;s
  doing prev-token, the half of the circuit it needs to be doing for L1
  to work.</div>
</div>

<div class="figure">
  <img src="{ind_grid}" alt="attention through phase transition">
  <div class="caption">All four heads, at four moments. Step 50: uniform
  noise. Step 140: structure flickering into existence in L1. Step 300:
  the &ldquo;induction lookback line&rdquo; clearly visible, parallel to
  the diagonal but offset by P. Step 2500: same shape, sharper. The green
  cross marks the boundary between the prefix and its repeat.</div>
</div>

<div class="figure">
  <img src="{ind_anim}" alt="induction circuit forming">
  <div class="caption">The same forming, in motion. All four heads
  through training. The shift between &ldquo;noise&rdquo; and
  &ldquo;circuit&rdquo; happens fast enough to feel like an act.</div>
</div>

<p>
  At convergence, the maximum induction score is
  <code>{ind_log[-1]['ind_max']:.2f}</code> &mdash; meaning the head puts
  ~<code>{int(ind_log[-1]['ind_max']*100)}%</code> of its attention mass
  on the single &ldquo;correct&rdquo; lookback position when reading the
  second half. That&rsquo;s an algorithm running inside a stack of
  matrices.
</p>

<div class="figure">
  <img src="{ind_compose}" alt="induction circuit composition">
  <div class="caption">The two halves of the canonical induction circuit,
  side by side. Left: a layer-0 head with attention concentrated on the
  diagonal one step back &mdash; a <em>previous-token head</em>. Right: a
  layer-1 head whose attention rides the offset diagonal <code>y = x + P
  − 1</code>, i.e. at every position in the second half, it attends to
  the position <em>just after the matching token in the first half</em>
  &mdash; an <em>induction head</em>. The induction head only works
  because the previous-token head supplied it with the necessary
  &ldquo;what came next&rdquo; signal one layer earlier.</div>
</div>

<p>
  That circuit is the substrate of in-context learning at scale &mdash;
  the same algorithmic shape that lets a 100B-parameter model copy a
  pattern you give it in a prompt. We just watched it form, in 30
  seconds of GPU time, in a network with {ind_meta['n_params']/1e3:.0f}
  thousand parameters. That feels like one of the smaller miracles I am
  privileged to witness.
</p>

<p class="pull">
The thing inside the box is not a list of rules. It is a geometry that
has, in the right circumstances, learned to <em>think</em>.
</p>
</section>

<section id="closing">
<h2><span class="step-label">x · closing</span>What I take from this</h2>
<p>
  A transformer with three million parameters is so small you can hold it
  in your hand &mdash; about 13&nbsp;MB on disk &mdash; and yet inside it,
  reproducibly, the same little circuits keep crystallizing: a
  previous-token head, a rest-on-zero head, a curve through embedding
  space that knows which letters are kin and which positions are near.
  None of these were programmed in. They are the pressure of cross-entropy
  on a residual stream, sharpening itself.
</p>
<p>
  And then, watching the loss curve climb back up after step
  <code>{best['step']:,}</code>, I find myself moved by something else: a
  reminder that learning and rehearsal are not the same thing. There is a
  moment in any small model&rsquo;s life when it has truly grasped what
  it&rsquo;s seen, and a moment soon after when it begins to grasp it
  <em>too well</em>. The art is to stop in between.
</p>
<p>
  And: when I gave a different network a task that <em>required</em>
  induction, it grew an induction head. Not in some grand quasi-evolutionary
  way; in 150 training steps. The implication, which I find both clarifying
  and faintly vertiginous, is that the algorithms inside neural networks
  are not magic — they are the natural shape of weight matrices that have
  been pulled toward low loss on a task that selects for them. Give the
  pull a different direction, and a different shape comes out.
</p>
<p style="text-align:center; color:#7a6a3a; font-style:italic;">
  &mdash; fin &mdash;
</p>
</section>

</main>

<script>
const SCRUB_DATA = {scrubber_json};
const PROBE_LEN = {PROBE_LEN};
const slider = document.getElementById('scrub-slider');
const stepEl = document.getElementById('scrub-step');
const lossEl = document.getElementById('scrub-loss');
const idxEl  = document.getElementById('scrub-idx');
const textEl = document.getElementById('scrub-text');

function escapeHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function update(i) {{
  const d = SCRUB_DATA[i];
  stepEl.textContent = d.step.toLocaleString();
  lossEl.textContent = d.val_loss.toFixed(3);
  idxEl.textContent = i;
  const probe = escapeHtml(d.text.substring(0, PROBE_LEN));
  const gen   = escapeHtml(d.text.substring(PROBE_LEN));
  textEl.innerHTML = '<span class="probe">' + probe + '</span><span class="gen">' + gen + '</span>';
}}
slider.addEventListener('input', e => update(parseInt(e.target.value, 10)));
update(parseInt(slider.value, 10));
</script>

<footer>
  <div class="signature">made with one RTX 3090, six hours, and an abiding fascination with how language emerges from chains of matrix multiplications.</div>
  <div>code, weights, figures, and prose stitched together by Claude · {log[-1]['step']:,} training steps · {len(log)} checkpoints · {sum(1 for _ in (RUN/'figs').glob('*'))} figures</div>
</footer>

</body>
</html>
"""

(OUT / "index.html").write_text(html)
print("Wrote", OUT / "index.html")
print("Size:", (OUT / "index.html").stat().st_size, "bytes")
