"""
Court Interpreter — Sight-Translation Scorer (demo)

A Gradio wrapper over the `evaleng` engine. Two ways in:
  • type a Spanish rendering, or
  • upload / record Spanish audio -> Whisper ASR -> transcript + delivery metrics,
and both produce the same per-unit scoring report. Runs the real engine.
"""
import os
import sys
import glob

# The engine is vendored under ./src ; make it importable without installing.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import gradio as gr

from evaleng.ingest.loader import load_fixture
from evaleng.interfaces import CandidateInput
from evaleng.pipeline import score
from evaleng.delivery import delivery_metrics

FIXTURE_DIR = os.path.join(HERE, "fixtures")
HEADERS = ["Unit", "Type", "Verdict", "Why", "Source span", "Candidate said"]

# Verdict shown as a solid colored pill (rendered in the "markdown" Verdict column).
def _pill(text, bg, big=False):
    pad = "4px 14px" if big else "2px 10px"
    fs = "14px" if big else "12px"
    return (f'<span class="verdict-pill" style="display:inline-block;padding:{pad};'
            f'border-radius:999px;font-size:{fs};font-weight:600;letter-spacing:.03em;'
            f'color:#ffffff;background:{bg}">{text}</span>')

STATUS = {
    "pass":     _pill("PASS", "#16a34a"),
    "fail":     _pill("FAIL", "#dc2626"),
    "unscored": _pill("UNSCORED", "#6b7280"),
}
# Column render types: the Verdict column (now index 2) is markdown so the pill HTML renders.
COL_TYPES = ["str", "str", "markdown", "str", "str", "str"]

# Slightly larger base font across the whole page.
THEME = gr.themes.Default(text_size=gr.themes.sizes.text_lg)

# Page CSS: keep the report table at its original size (the theme enlarges the
# rest of the page).
CSS = """
#report, #report * { font-size: 13px !important; }
#report .verdict-pill { font-size: 12px !important; }

/* Make the input tabs read more like classic folder tabs */
.tab-nav { border-bottom: 2px solid var(--border-color-primary) !important; gap: 2px !important; }
.tab-nav button { border: 1px solid transparent !important; border-bottom: none !important;
                  border-radius: 8px 8px 0 0 !important; padding: 8px 18px !important;
                  margin-bottom: -2px !important; }
.tab-nav button.selected { background: var(--background-fill-primary) !important;
                           border-color: var(--border-color-primary) !important;
                           border-bottom: 2px solid var(--background-fill-primary) !important;
                           font-weight: 600 !important; color: #2563eb !important; }
"""


def _label_for(path, fx):
    """A readable dropdown label, e.g. 'st_en_es_001 — Case Narrative'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    desc = " ".join(parts[4:]).title() if len(parts) > 4 else ""
    return f"{fx.id} — {desc}" if desc else fx.id


def _load_fixtures():
    """Load the real sight-translation fixtures (skips the rough reference_draft)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(FIXTURE_DIR, "st_en_es_*.yaml"))):
        try:
            fx = load_fixture(path)
            out[_label_for(path, fx)] = fx
        except Exception as e:
            print(f"[fixtures] skipped {os.path.basename(path)}: {e}")
    return out


FIXTURES = _load_fixtures()
LABELS = list(FIXTURES.keys())
if not LABELS:
    raise SystemExit("No fixtures found in ./fixtures")

# Warm Stanza at startup; Whisper is loaded lazily on first audio use.
try:
    from evaleng.analysis import pipeline as _warm_stanza
    _warm_stanza()
    print("[startup] Stanza Spanish model ready.")
except Exception as e:
    print(f"[startup] Stanza not warmed yet (loads on first score): {e}")


def _needs_policy(u):
    """Grammar units need a feature_spec; register units need a register_spec.
    Units missing these can't be scored yet, so we skip them gracefully rather
    than let one abort the whole report."""
    mp = u.matching_policy
    if u.type.value == "grammar":
        return mp is None or mp.feature_spec is None
    if u.type.value == "register":
        return mp is None or mp.register_spec is None
    return False


def _score_candidate(fx, cand):
    """Score a CandidateInput (from typed text or from ASR) -> (summary_md, rows)."""
    units = list(fx.units)
    scorable = [u for u in units if not _needs_policy(u)]
    deferred = [u for u in units if _needs_policy(u)]
    span = {u.id: u.source_span for u in units}

    report = score(cand, fx.model_copy(update={"units": scorable}))
    r = report.result
    rows = [
        [v.unit_id, v.unit_type.value, STATUS.get(v.status, v.status), v.reason,
         span.get(v.unit_id, ""), v.candidate_text or "—"]
        for v in r.verdicts
    ]
    for u in deferred:
        rows.append([u.id, u.type.value, STATUS["unscored"],
                     "needs a matching policy (feature_spec / register_spec) — not yet specified",
                     u.source_span, "—"])
    order = {u.id: i for i, u in enumerate(units)}
    rows.sort(key=lambda row: order.get(row[0], len(order)))

    achieved = (r.weighted_passed / r.weighted_scored) if r.weighted_scored else 0.0
    scored, _ = r.coverage
    verdict = (_pill("PASS", "#16a34a", big=True) if r.overall_pass
               else _pill("FAIL", "#dc2626", big=True))
    note = (f"\n- {len(deferred)} unit(s) not yet scorable (missing a matching policy)"
            if deferred else "")
    summary = (
        f"**Overall:** {verdict}\n\n"
        f"- Passed **{r.weighted_passed} / {r.weighted_scored}** scored units "
        f"= **{achieved:.0%}**  (needs {r.pass_ratio:.0%})\n"
        f"- Coverage: {scored} / {len(units)} units scored{note}"
    )
    return summary, rows


def run_score(label, rendering):
    """Score a typed candidate rendering."""
    fx = FIXTURES[label]
    text = (rendering or "").strip()
    if not text:
        return "**Enter a candidate rendering above, then press Score.**", []
    try:
        cand = CandidateInput(text=text, tokens=text.split())
        return _score_candidate(fx, cand)
    except Exception as e:
        return f"**Could not score this rendering.**\n\n`{type(e).__name__}: {e}`", []


def _delivery_md(m):
    """Render DeliveryMetrics (or None) as a small markdown block."""
    if m is None:
        return "*No delivery metrics (the audio had no word timings).*"
    rate = f"**{m.words_per_minute} wpm**" if m.words_per_minute else "n/a"
    fillers = f" ({', '.join(m.hesitations)})" if m.hesitations else ""
    audio = f"{m.audio_seconds}s audio · " if m.audio_seconds else ""
    return (
        "#### 🎧 Delivery (not scored)\n"
        f"- {m.words} words · {m.speech_seconds}s speech · {audio}rate {rate}\n"
        f"- Long pauses (> 2s): **{len(m.long_pauses)}**\n"
        f"- Hesitations: **{len(m.hesitations)}**{fillers}\n"
        f"- False starts (approx): **{m.false_starts}**"
    )


def run_audio(label, audio_path, model_size, nbest):
    """Transcribe audio with Whisper, score it, and report delivery metrics.

    Returns (summary_md, transcript, transcript_for_textbox, table_rows, delivery_md)."""
    if not audio_path:
        return ("**Upload or record Spanish audio, then press Transcribe & Score.**",
                "", "", [], "")
    fx = FIXTURES[label]
    try:
        from evaleng.asr import transcribe          # lazy: Whisper only loaded on audio use
        cand = transcribe(audio_path, size=model_size, n_alternatives=int(nbest))
    except Exception as e:
        return f"**Transcription failed.**\n\n`{type(e).__name__}: {e}`", "", "", [], ""

    delivery = _delivery_md(delivery_metrics(cand))
    nbest_note = (f"\n- ASR n-best: **{len(cand.nbest)}** alternative(s) captured for rescue"
                  if cand.nbest else "")
    try:
        summary, rows = _score_candidate(fx, cand)
    except Exception as e:
        return (f"**Scoring failed.**\n\n`{type(e).__name__}: {e}`",
                cand.text, cand.text, [], delivery)
    return summary + nbest_note, cand.text, cand.text, rows, delivery


INTRO = """
# Court Interpreter Sight Translation Scorer V0.1 (demo)

A live demo of **`evaleng`**, an evaluation engine that scores a candidate interpreter's rendering like a certification exam. It evaluates against a fixed inventory of scoring units: specific terms, numbers, grammatical features, and register. The engine combines classical, statistical, neural NLP techniques and generates a deterministic span-level feedback with rule-based scoring logic and no LLM. It is the evaluation core of a larger web-based court-interpreter training tool.

**Two ways to try it:**
- **Type a rendering** — enter the Spanish text and press *Score*.
- **Upload / record audio** — Whisper transcribes it (ASR), it's scored, and
  delivery metrics (pace, long pauses, hesitations) are shown alongside.

Pick an exam text, then use either tab below.
"""


def on_fixture(label):
    """Show the chosen text's source (reference stays hidden in the accordion),
    pre-fill the candidate box with the reference, and clear prior results."""
    fx = FIXTURES[label]
    ref = fx.reference_rendering.strip()
    return (
        gr.update(value=fx.source_text.strip()),   # source
        gr.update(value=ref),                       # reference (inside accordion)
        gr.update(value=""),                        # candidate box (start empty)
        gr.update(value=""),                        # summary
        gr.update(value=[]),                        # table
        gr.update(value=""),                        # delivery
        gr.update(value=""),                        # transcript
    )


def build():
    with gr.Blocks(title="Court Interpreter Sight-Translation Scorer", theme=THEME, css=CSS) as demo:
        gr.Markdown(INTRO)

        first = FIXTURES[LABELS[0]]
        fixture_dd = gr.Dropdown(LABELS, value=LABELS[0], label="Exams")
        source_tb = gr.Textbox(value=first.source_text.strip(),
                               label="English source", interactive=False, lines=3)

        # Reference rendering is hidden by default — expand to reveal the "answer".
        with gr.Accordion("Reference rendering (Spanish) — click to reveal", open=False):
            reference_tb = gr.Textbox(value=first.reference_rendering.strip(),
                                      label="Reference rendering", show_label=False,
                                      interactive=False, lines=3)

        with gr.Tabs():
            with gr.Tab("🎤 Upload / record audio"):
                audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                    label="Spanish audio (the candidate's spoken rendering)")
                with gr.Row():
                    model_dd = gr.Dropdown(["tiny", "base", "small"], value="tiny",
                                           label="Whisper model (larger = slower, more accurate)")
                    nbest_sl = gr.Slider(0, 3, value=0, step=1,
                                         label="ASR n-best alternatives (fuzzy rescue)")
                audio_btn = gr.Button("Transcribe & Score", variant="primary")
                transcript_tb = gr.Textbox(label="Recognized transcript (Whisper ASR)",
                                           interactive=False, lines=4)
                delivery_md = gr.Markdown()
            with gr.Tab("✏️ Type a rendering"):
                candidate_tb = gr.Textbox(value="",
                                          placeholder="Type or paste the Spanish rendering here, then press Score…",
                                          label="Candidate rendering",
                                          interactive=True, lines=5)
                score_btn = gr.Button("Score", variant="primary")

        summary_md = gr.Markdown()
        table = gr.Dataframe(headers=HEADERS, datatype=COL_TYPES, wrap=True,
                             interactive=False, label="Per-unit report", elem_id="report")

        fixture_dd.change(
            on_fixture, inputs=fixture_dd,
            outputs=[source_tb, reference_tb, candidate_tb, summary_md, table,
                     delivery_md, transcript_tb],
        )
        score_btn.click(run_score, inputs=[fixture_dd, candidate_tb],
                        outputs=[summary_md, table])
        audio_btn.click(run_audio, inputs=[fixture_dd, audio_in, model_dd, nbest_sl],
                        outputs=[summary_md, transcript_tb, candidate_tb, table, delivery_md])
    return demo


# Module-level `demo` so a host that imports this file finds a `demo` object.
demo = build()

if __name__ == "__main__":
    # share=True publishes a temporary public *.gradio.live link that proxies to
    # this locally-running app — reachable by anyone while your machine runs it.
    # Set GRADIO_SHARE=false to run local-only.
    share = os.environ.get("GRADIO_SHARE", "true").lower() in ("1", "true", "yes")
    demo.launch(share=share, inbrowser=False)
