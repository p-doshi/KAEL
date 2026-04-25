"""
KAEL Terminal Interface — Phase 0 / Phase 1
Commands: /stats /tau /eval /recent /phase1 /alpha /inject /graph /quit /help
"""
import os                                      # ← ADD THIS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # ← AND THIS
import torch
if torch.cuda.is_available():
    try:
        torch.zeros(1, device="cuda"); torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"CUDA broken: {e}\nFix: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm")
        raise SystemExit(1)
import sys
import logging
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from config import cfg
from core.model import KAELModel
from core.runner import SessionRunner
from memory.session_store import SessionStore
from memory.crawler_processor import CrawlProcessor
from core.counsellor import KAELCounsellor

console = Console() if HAS_RICH else None

BANNER = """
╔═══════════════════════════════════════╗
║  KAEL — Kinetic Adaptive Epistemic    ║
║         Lattice  [Phase 0/1]          ║
╚═══════════════════════════════════════╝
  Commands: /stats  /tau  /phase1  /alpha  /graph  /crawl  /quit  /help
"""

HELP_TEXT = """
Available commands:
  /stats      — session database statistics
  /tau        — current tau vector info (norm, sub-embedding norms)
  /eval       — run Phase 0 benchmark suite
  /recent     — show last 5 sessions
  /phase1     — activate Phase 1 (tau injection into attention)
  /alpha N    — set gravitational pull strength  e.g. /alpha 0.2
  /inject     — show attention hook stats
  /graph      — open tau_relational knowledge graph in browser
  /crawl      — fetch frontier feeds and digest (arxiv cs.AI/LG/CL, quanta, etc)
  /crawl feed <url>   — add and crawl a specific feed URL
  /crawl url <url>    — crawl a specific page
  /crawl search <q>   — search arxiv for query and process results
  /crawl status       — show crawl cache stats
  /counsel <question> — ask Perplexity for a second perspective
  /counsel status     — show counsellor stats
  /autonomous start   — start overnight learning loop (ctrl+c to stop)
  /autonomous status  — show loop state
  /autonomous stop    — stop the loop
  /quit       — exit
  /help       — this message

Everything else is sent to KAEL as a prompt.
"""


def print_banner():
    if HAS_RICH:
        console.print(Panel(BANNER.strip(), border_style="dim"))
    else:
        print(BANNER)


def print_stats(store):
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


def print_tau(model):
    tau = model.tau
    tau_vec = tau.tau.detach()
    e_norm = tau_vec[tau.epistemic_slice].norm().item()
    d_norm = tau_vec[tau.dispositional_slice].norm().item()
    r_norm = tau_vec[tau.relational_slice].norm().item()
    total_norm = tau_vec.norm().item()

    if HAS_RICH:
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_row("[dim]total norm[/dim]", f"{total_norm:.6f}")
        t.add_row("[dim]tau_epistemic norm[/dim]", f"{e_norm:.6f}")
        t.add_row("[dim]tau_dispositional norm[/dim]", f"{d_norm:.6f}")
        t.add_row("[dim]tau_relational norm[/dim]", f"{r_norm:.6f}")
        t.add_row("[dim]phase[/dim]", str(model._phase))
        inj = model.injection_stats()
        if inj.get("active"):
            t.add_row("[dim]alpha[/dim]", str(inj.get("alpha")))
            t.add_row("[dim]hook calls[/dim]", str(inj.get("total_hook_calls")))
        console.print(Panel(t, title="[bold]tau State[/bold]", border_style="dim"))
    else:
        print(f"  total norm:    {total_norm:.6f}")
        print(f"  epistemic:     {e_norm:.6f}")
        print(f"  dispositional: {d_norm:.6f}")
        print(f"  relational:    {r_norm:.6f}")
        print(f"  phase:         {model._phase}")


def print_recent(store):
    sessions = store.get_recent_sessions(5)
    for s in reversed(sessions):
        if HAS_RICH:
            gate_str = f" · g={s.gate_value:.3f}" if s.gate_value is not None else ""
            novelty_str = f" · novelty={s.novelty_score:.2f}" if s.novelty_score is not None else ""
            console.print(f"\n[dim]{s.domain or 'general'}{novelty_str}{gate_str} · {s.session_id[:8]}[/dim]")
            console.print(f"[bold]You:[/bold] {s.user_input[:120]}{'...' if len(s.user_input) > 120 else ''}")
            console.print(f"[bold]KAEL:[/bold] {s.model_output[:200]}{'...' if len(s.model_output) > 200 else ''}")
        else:
            print(f"\n[{s.domain}] You: {s.user_input[:80]}")
            print(f"KAEL: {s.model_output[:120]}")


def run_repl(runner, store, model):
    print_banner()
    _processor = CrawlProcessor(runner, store)
    _counsellor = KAELCounsellor()
    _loop_instance = None

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
        elif user_input == "/phase1":
            if model._phase >= 1:
                print("Phase 1 already active.")
            else:
                model.activate_phase1()
                print(f"Phase 1 activated. alpha={model.config.tau.attention_alpha}")
        elif user_input.startswith("/alpha"):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    a = float(parts[1])
                    model.set_alpha(a)
                    print(f"alpha set to {a}")
                except ValueError:
                    print("Usage: /alpha 0.15")
            else:
                print("Usage: /alpha 0.15")
        elif user_input == "/inject":
            stats = model.injection_stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")
        elif user_input == "/graph":
            try:
                from interface.knowledge_graph import build_and_open_graph
                build_and_open_graph(store, model)
            except Exception as e:
                print(f"Graph error: {e}")
        elif user_input == "/crawl" or user_input.startswith("/crawl "):
            import asyncio
            parts = user_input.split(maxsplit=2)
            subcommand = parts[1] if len(parts) > 1 else "feeds"

            if subcommand == "feeds" or subcommand == "feed" and len(parts) == 2:
                print("Fetching frontier feeds (digest mode)...")
                result = asyncio.run(_processor.run_feeds(mode="digest"))
                print(f"Done: {result['pages']} pages -> {result['sessions']} sessions | domains: {result.get('domains', [])}")

            elif subcommand == "feed" and len(parts) == 3:
                url = parts[2]
                print(f"Fetching feed: {url}")
                result = asyncio.run(_processor.run_feeds(feeds=[url], mode="digest"))
                print(f"Done: {result['pages']} pages -> {result['sessions']} sessions")

            elif subcommand == "url" and len(parts) == 3:
                url = parts[2]
                print(f"Crawling: {url}")
                result = asyncio.run(_processor.run_urls([url], mode="per_page"))
                print(f"Done: {result['pages']} pages -> {result['sessions']} sessions")

            elif subcommand == "search" and len(parts) == 3:
                query = parts[2]
                print(f"Searching ArXiv: '{query}'")
                result = asyncio.run(_processor.run_query(query, mode="per_page"))
                print(f"Done: {result['pages']} results -> {result['sessions']} sessions")

            elif subcommand == "status":
                cache = _processor.crawler.cache_dir
                seen = len(_processor.crawler._seen_hashes)
                print(f"Crawl cache: {cache}")
                print(f"Seen content hashes: {seen}")
                print(f"Configured feeds: {len(_processor.crawler._robots_cache)} robots cached")

            else:
                print("Usage: /crawl | /crawl feed <url> | /crawl url <url> | /crawl search <query> | /crawl status")
        elif user_input.startswith("/counsel"):
            parts = user_input.split(maxsplit=1)
            subcommand = parts[1] if len(parts) > 1 else "status"

            if subcommand == "status":
                stats = _counsellor.stats()
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            elif not _counsellor.available:
                print("Counsellor inactive. Set PERPLEXITY_API_KEY environment variable.")
                print("  export PERPLEXITY_API_KEY=pplx-...")
            else:
                result = _counsellor.consult(subcommand, trigger="manual")
                if result:
                    if HAS_RICH:
                        console.print(f"\n[bold]Perplexity[/bold] [dim]({result.model} · {result.tokens_used} tokens · {result.elapsed}s)[/dim]")
                        console.print(result.response)
                    else:
                        print(f"\nPerplexity: {result.response}")
                    # Feed response into KAEL as a session for tau update
                    runner.run(
                        user_input=f"Perplexity's perspective on '{subcommand[:60]}':\n\n{result.response}",
                        capture_embedding=True,
                    )

        elif user_input.startswith("/autonomous"):
            import asyncio
            parts = user_input.split(maxsplit=1)
            subcommand = parts[1] if len(parts) > 1 else "status"

            if subcommand == "start":
                from memory.autonomous_loop import AutonomousLoop
                _loop_instance = AutonomousLoop(
                    runner=runner,
                    store=store,
                    counsellor=_counsellor,
                    max_storage_mb=500,
                    max_hours=8,
                    max_sessions=500,
                    cooldown_seconds=10,
                )
                print("Autonomous loop starting. Press Ctrl+C to stop.")
                print(f"Limits: 8h / 500 sessions / 500MB storage")
                print(f"Counsellor: {'active' if _counsellor.available else 'inactive (no API key)'}")
                try:
                    asyncio.run(_loop_instance.start())
                except KeyboardInterrupt:
                    print("\nLoop stopped by user.")
                    status = _loop_instance.status()
                    print(f"  cycles={status['cycle']} sessions={status['sessions_created']} pages={status['pages_processed']}")

            elif subcommand == "status":
                if _loop_instance:
                    s = _loop_instance.status()
                    for k, v in s.items():
                        print(f"  {k}: {v}")
                else:
                    print("No loop running. Use /autonomous start")

            elif subcommand == "stop":
                if _loop_instance:
                    _loop_instance.stop("manual")
                    print("Stop signal sent.")
                else:
                    print("No loop running.")
            else:
                print("Usage: /autonomous start | /autonomous status | /autonomous stop")

        elif user_input == "/eval":
            print("Running eval suite... (this will take a while)")
            from eval.phase0_eval import Phase0Eval
            evaluator = Phase0Eval(model)
            evaluator.run_all(phase=model._phase)
        else:
            try:
                result = runner.run(user_input)
                if HAS_RICH:
                    gate_str = f" · g={result['meta']['gate']:.3f}" if result['meta']['gate'] is not None else ""
                    flag_str = " · [bold red]FLAGGED[/bold red]" if result['meta']['flagged'] else ""
                    console.print(
                        f"\n[bold]KAEL[/bold] [dim]("
                        f"session {result['meta']['session_number']} · "
                        f"{result['meta']['domain']} · "
                        f"novelty={result['meta']['novelty']:.2f}"
                        f"{gate_str} · "
                        f"{result['meta']['elapsed']}s)[/dim]{flag_str}"
                    )
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
    
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"CUDA context broken: {e}")
            print("Fix: sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia")
            sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, cfg.logging.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(cfg.logging.log_dir / "kael.log"),
            logging.StreamHandler(sys.stdout),
        ]
    )

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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()