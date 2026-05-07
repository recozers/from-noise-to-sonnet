"""
Generate HTML for the neuron-viewer section.
Reads run/analysis/neurons.json, renders a curated set with
token-level activation highlighting, and writes
run/analysis/neuron_viewer.html (an HTML fragment to embed in the report).
"""
import json, re
from pathlib import Path

ROOT = Path("/home/stuart/sonnet_mind")
RUN  = ROOT / "run"
ANA  = RUN / "analysis"

records = json.loads((ANA / "neurons.json").read_text())

# Hand-curated picks: the cleanest interpretable neurons + my best-guess label
CURATED = [
    # (layer, neuron, label, blurb)
    (1, 664, "the all-caps speaker-header detector",
     "fires on letters inside the model's representation of an ALL-CAPS character name at the start of a line of dialogue. Look at the K=8 examples — every one is a letter inside something like 'HENRY BOLINGBROKE', 'PETRUCHIO', 'ARCHBISHOP OF YORK'."),
    (1, 867, "proper-noun-after-of",
     "fires on a capital letter that follows the word 'of'. Every top activation is the model recognising 'Duke of <Capital>' or 'Earl of <Capital>' — a preposition-and-proper-noun bigram detector."),
    (1, 224, "the end-of-utterance punctuation neuron",
     "fires on '!' and '?' tokens. Crucially, not on '.' — the model has separated emphatic punctuation from neutral. A real semantic distinction, learned from cross-entropy alone."),
    (1, 765, "the 'name ends in -IUS:' neuron",
     "fires on the S character at the end of a Roman/Latinate speaker name immediately preceding the colon. MENENIUS, MAMILLIUS, &c. The model has discovered a small-but-real feature of Plutarchian Shakespeare."),
    (0, 566, "the new-speaker-coming neuron",
     "fires on the newline character that precedes an ALL-CAPS character name on its own line. A 'a new speaker is about to talk' detector."),
    (0, 895, "mid-sentence punctuation",
     "fires on commas, semicolons, periods inside sentences — but not at the end of paragraphs. A 'pause-not-stop' detector."),
    (0, 457, "soft break — newlines and spaces",
     "fires on whitespace separating words and lines. The most basic 'segment boundary' feature."),
    (0, 198, "mid-name capital letter (U/I)",
     "fires on uppercase vowels inside speaker headers — CORIOLAN<b>U</b>S, &c. A smaller, narrower cousin of L1.N664."),
    (1, 671, "the dash-and-restart neuron",
     "fires after em-dashes ('--') and at the start of new sentences. A 'we're starting a new clause' detector."),
    (2, 965, "the 'scr-' onset neuron",
     "fires on the R inside 'scr' and similar consonant onsets — 'prescription', 'scroll', 'scrivener', 'bestrew'. A specific letter-cluster detector."),
]

by_id = {(r["layer"], r["neuron"]): r for r in records}

def color_for(act, peak):
    """Return a CSS background-color for a given activation."""
    if peak <= 0: return "transparent"
    intensity = max(0.0, min(1.0, act / peak))
    # Gold-ish gradient
    if intensity < 0.05: return "transparent"
    # alpha from intensity
    a = intensity * 0.85 + 0.05
    return f"rgba(233,180,76,{a:.3f})"

def render_snippet(s, peak):
    """Render one snippet with per-character highlighting."""
    chars = s["chars"]
    acts = s["act_per_char"]
    hl   = s["highlight_pos"]
    out = []
    for i, (c, a) in enumerate(zip(chars, acts)):
        # display char
        if c == "\n":
            disp = "↵"
        elif c == " ":
            disp = "·"
        else:
            disp = c.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        bg = color_for(a, peak)
        cls = "hl" if i == hl else ""
        out.append(f'<span class="ch {cls}" style="background:{bg}">{disp}</span>')
    return f'<div class="snippet"><span class="actval">{s["act"]:.2f}</span>{"".join(out)}</div>'

cards = []
for L, N, label, blurb in CURATED:
    rec = by_id.get((L, N))
    if rec is None: continue
    snippet_html = "\n".join(render_snippet(s, rec["peak"]) for s in rec["snippets"][:6])
    cards.append(f"""
<div class="neuron-card">
  <div class="neuron-head">
    <span class="neuron-id">L{L}.N{N}</span>
    <span class="neuron-label">{label}</span>
    <span class="neuron-peak">peak {rec["peak"]:.2f}</span>
  </div>
  <div class="neuron-blurb">{blurb}</div>
  <div class="snippets">{snippet_html}</div>
</div>
""")

html = '<div class="neuron-viewer">\n' + "\n".join(cards) + "\n</div>"
(ANA / "neuron_viewer.html").write_text(html)
print(f"Wrote {len(cards)} neuron cards -> {ANA/'neuron_viewer.html'}")
print(f"Size: {len(html)} bytes")
