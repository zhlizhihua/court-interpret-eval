# court-interpret-eval

**An automated evaluation engine for court-interpreter sight translation.**

`evaleng` grades a candidate interpreter's rendering like a certification exam — against a fixed inventory of **scoring units** (specific terms, numbers, grammatical features, and register). The engine uses classical + statistical + neural NLP techniques and generates a span-level deterministic feedback. It is the evaluation core of a larger web-based court-interpreter training tool.

---


## Getting started

The environment is managed with conda:

```bash
conda env create -f environment.yml
conda activate evaleng
pip install -e .            # installs the src-layout package in editable mode
python scripts/verify_toolchain.py   # confirms Stanza, RapidFuzz, Whisper, etc. load
```

Score directly from audio:

```bash
python -m evaleng.cli \
    --fixture fixtures/reference_draft.yaml \
    --audio path/to/candidate.mp3 \
    --verbose
```

Score a transcript:

```bash
python -m evaleng.cli \
    --fixture fixtures/reference_draft.yaml \
    --transcript path/to/candidate.txt
```

Score directly from audio, sampling extra ASR hypotheses for n-best rescue:

```bash
python -m evaleng.cli \
    --fixture fixtures/reference_draft.yaml \
    --audio path/to/candidate.mp3 \
    --asr-model small \
    --nbest 3 \
    --verbose
```

## Testing

```bash
pytest                 # full suite
pytest -m "not slow"   # skip Stanza-dependent tests
```

---

## Layout

```
src/evaleng/
├── schema/models.py     Pydantic scoring-unit + fixture schema
├── ingest/loader.py     YAML → validated fixture
├── interfaces.py        typed contracts passed between stages
├── dispatch.py          unit-type → matcher routing
├── localize.py          reference-projected span localization
├── match/ 
│   ├── number.py        text2num spoken-form normalization → exact compare
│   ├── lexical.py       Stanza lemmatization + RapidFuzz fuzzy matching
│   ├── grammar.py       Universal Dependencies feature checks
│   └── register.py      coarse classifier over Stanza morphology
├── analysis.py          shared Stanza pipeline
├── asr.py               Whisper transcription
├── delivery.py          delivery metrics from word timings
├── pipeline.py          localize → match → n-best → aggregate → feedback
└── cli.py               command-line entry point
fixtures/                exam fixtures (source, reference, units, threshold)
scripts/                 toolchain smoke test
tests/                   pytest suite
```