# KAEL — Phase 0

**Kinetic Adaptive Epistemic Lattice**

Phase 0 establishes the infrastructure everything else builds on:
frozen base model, session logging, τ parameter, and baseline benchmarks.
τ does not yet modulate inference — that is Phase 1.

---

## Setup

```bash
cd kael/
python scripts/setup.py
```

This installs dependencies, verifies CUDA, creates directories, and runs tests.

## First run

```bash
python interface/repl.py
```

Commands inside the REPL:
- `/stats` — session database statistics
- `/tau` — current τ state (norm, sub-embedding breakdown)
- `/eval` — run Phase 0 benchmark suite (GSM8K, MMLU, Frontier Q&A, Consistency)
- `/recent` — last 5 sessions
- `/quit` — exit

## Config

Edit `config.py` before running:

```python
# For smaller GPUs, change base model:
model_name: str = "Qwen/Qwen2.5-7B-Instruct"   # ~14GB VRAM
# or
load_in_4bit: bool = True                        # halves VRAM at small quality cost

# τ dim must match model's hidden_size:
# Qwen2.5-3B  → dim=2048
# Qwen2.5-7B  → dim=3584
# Qwen2.5-14B → dim=5120
```

**Important:** If you change the base model, update `tau.dim` and the
three sub-embedding dims to sum to the new value.

## Project structure

```
kael/
├── config.py                  # All hyperparameters
├── core/
│   ├── model.py               # KAELModel = frozen θ + τ
│   └── runner.py              # SessionRunner — inference + logging
├── memory/
│   └── session_store.py       # SQLite session DB
├── eval/
│   └── phase0_eval.py         # Baseline benchmarks
├── interface/
│   └── repl.py                # Terminal REPL
├── tests/
│   └── test_phase0.py         # Unit tests
├── scripts/
│   └/setup.py                 # One-time setup
├── logs/                      # Session logs, eval results
├── checkpoints/               # τ snapshots
└── memory/                    # sessions.db, FAISS index (Phase 4)
```

## Phase 0 checklist

- [ ] `python scripts/setup.py` — all tests pass
- [ ] `python interface/repl.py` — model loads, REPL responds
- [ ] `/tau` — τ norm is non-zero (small, ~0.01-0.05 range)
- [ ] 5 manual sessions logged — `/stats` shows count=5
- [ ] `/eval` — baseline scores recorded in `logs/eval/phase0_results.json`

When all checked: Phase 0 complete. Move to Phase 1 (τ injection into attention).

## Phase 0 → Phase 1 transition

The one line that activates Phase 1:
```python
model.tau.activate_phase1(model.get_model_hidden_size())
```

Everything else in the architecture is already wired.
The soft prefix projections initialize near-zero so early Phase 1
behavior is almost identical to Phase 0 — τ effect grows as it's trained.

---

*Identity is not a starting state. It is the shape of a trajectory.*
