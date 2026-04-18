"""
KAEL Terminal Interface
Simple REPL for Phase 0 development and testing.
Rich-formatted output. Session stats on demand.
Commands: /stats /tau /eval /quit /help
"""

import sys
import logging
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from config import cfg
from core.model import KAELModel
from core.runner import SessionRunner
from memory.session_store import SessionStore

console = Console() if HAS_RICH else None

BANNER = """
╔═══════════════════════════════════════╗
║  KAEL — Kinetic Adaptive Epistemic    ║
║         Lattice  [Phase 0]            ║
╚═══════════════════════════════════════╝
  Commands: /stats  /tau  /eval  /quit  /help
"""

HELP_TEXT = """
Available commands:
  /stats    — session database statistics
  /tau      — current τ vector info (norm, sub-embedding norms)
  /eval     — run Phase 0 benchmark suite
  /recent   — show last 5 sessions
  /quit     — exit
  /help     — this message

Everything else is sent to KAEL as a prompt.
"""


def print_banner():
    if HAS_RICH:
        console.print(Panel(BANNER.strip(), border_style="dim"))
    else:
        print(BANNER)


def print_stats(store: SessionStore):
    stats = store.get_stats()
    if HAS_RICH:
        t = Table(show_header=False, box=None, padding=(0, 2))
        for k, v in stats.items():
            if k != "domain_distribution":
                t.add_row(f"[dim]{k}[/dim]", str(v))
        console.print(Panel(t, title="[bold]Session Stats[/bold]", border_style="dim"))
        if stats["domain_distribution"]:
            dt = Table("domain", "count", box=None, padding=(0, 2))
            for d, c in stats["domain_distribution"].items():
                dt.add_row(d, str(c))
            console.print(Panel(dt, title="[bold]Domains[/bold]", border_style="dim"))
    else:
        for k, v in stats.items():
            print(f"  {k}: {v}")


def print_tau(model: KAELModel):
    import torch
    tau = model.tau
    tau_vec = tau.tau.detach()

    e_norm = tau_vec[tau.epistemic_slice].norm().item()
    d_norm = tau_vec[tau.dispositional_slice].norm().item()
    r_norm = tau_vec[tau.relational_slice].norm().item()
    total_norm = tau_vec.norm().item()

    if HAS_RICH:
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_row("[dim]total norm[/dim]", f"{total_norm:.6f}")
        t.add_row("[dim]τ_epistemic norm[/dim]", f"{e_norm:.6f}")
        t.add_row("[dim]τ_dispositional norm[/dim]", f"{d_norm:.6f}")
        t.add_row("[dim]τ_relational norm[/dim]", f"{r_norm:.6f}")
        t.add_row("[dim]phase[/dim]", str(0))
        console.print(Panel(t, title="[bold]τ State[/bold]", border_style="dim"))
    else:
        print(f"  total norm: {total_norm:.6f}")
        print(f"  epistemic:  {e_norm:.6f}")
        print(f"  dispositional: {d_norm:.6f}")
        print(f"  relational: {r_norm:.6f}")


def print_recent(store: SessionStore):
    sessions = store.get_recent_sessions(5)
    for s in reversed(sessions):
        if HAS_RICH:
            console.print(f"\n[dim]{s.domain or 'general'} · {s.session_id[:8]}[/dim]")
            console.print(f"[bold]You:[/bold] {s.user_input[:120]}{'...' if len(s.user_input) > 120 else ''}")
            console.print(f"[bold]KAEL:[/bold] {s.model_output[:200]}{'...' if len(s.model_output) > 200 else ''}")
        else:
            print(f"\n[{s.domain}] You: {s.user_input[:80]}")
            print(f"KAEL: {s.model_output[:120]}")


def run_repl(runner: SessionRunner, store: SessionStore, model: KAELModel):
    print_banner()

    while True:
        try:
            if HAS_RICH:
                user_input = console.input("\n[bold cyan]you ›[/bold cyan] ").strip()
            else:
                user_input = input("\nyou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        # Commands
        if user_input == "/quit":
            break
        elif user_input == "/help":
            print(HELP_TEXT)
        elif user_input == "/stats":
            print_stats(store)
        elif user_input == "/tau":
            print_tau(model)
        elif user_input == "/recent":
            print_recent(store)
        elif user_input == "/eval":
            print("Running eval suite... (this will take a while)")
            from eval.phase0_eval import Phase0Eval
            evaluator = Phase0Eval(model)
            evaluator.run_all(phase=0)
        else:
            # Regular session
            try:
                result = runner.run(user_input)
                if HAS_RICH:
                    console.print(f"\n[bold]KAEL[/bold] [dim](session {result['meta']['session_number']} · {result['meta']['domain']} · {result['meta']['elapsed']}s)[/dim]")
                    console.print(result["response"])
                else:
                    print(f"\nKAEL [{result['meta']['session_number']}]: {result['response']}")
            except Exception as e:
                logging.exception(f"Session error: {e}")
                print(f"Error: {e}")


def main():
    import torch
    import random
    import numpy as np

    logging.basicConfig(
        level=getattr(logging, cfg.logging.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(cfg.logging.log_dir / "kael.log"),
            logging.StreamHandler(sys.stdout),
        ]
    )

    # Reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print("Loading KAEL...")
    model = KAELModel(cfg)
    store = SessionStore()
    runner = SessionRunner(model, store)

    run_repl(runner, store, model)


if __name__ == "__main__":
    main()
