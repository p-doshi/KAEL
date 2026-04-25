"""
KAEL Autonomous Learning Loop
KAEL runs overnight, deciding what to learn next, iteratively.

Flow:
  1. Fetch frontier feeds (seed)
  2. Process pages -> tau updates
  3. Assess what was most novel
  4. Decide next search query (curiosity engine or counsellor)
  5. Search ArXiv / fetch pages
  6. Repeat until: storage limit hit / time limit / novelty dried up

Safety constraints:
  - max_storage_mb: hard cap on DB + crawl cache size
  - max_hours: wall-clock time limit
  - max_sessions: session count limit
  - min_novelty_threshold: stop if nothing novel for N consecutive cycles
  - cooldown_seconds: min time between cycles (be nice to servers)

Curiosity engine:
  - Tracks which domains have highest tau novelty contribution
  - Generates search queries by combining high-novelty domain terms
  - Falls back to counsellor (Claude API) for direction if stuck
  - Avoids re-searching same queries (seen_queries set)

Run:
  python -m memory.autonomous_loop          # foreground with live output
  nohup python -m memory.autonomous_loop &  # background overnight
  /autonomous start  in REPL               # from within KAEL
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # set before torch loads
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import asyncio
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def _check_cuda_health() -> bool:
    import torch
    try:
        if not torch.cuda.is_available():
            return False
        t = torch.zeros(1, device="cuda")
        del t
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        logger.error(f"CUDA context unhealthy: {e}")
        logger.error("Fix: sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia")
        return False

# ── Curiosity query templates ──────────────────────────────────────────────────
# These seed the query generator when no counsellor is available.
# Domain -> list of frontier-oriented query fragments.

CURIOSITY_TEMPLATES = {
    "machine_learning": [
        "persistent memory transformers",
        "continual learning without forgetting",
        "emergent capabilities large language models",
        "mechanistic interpretability attention",
        "world models reinforcement learning",
        "meta-learning few-shot",
        "mixture of experts sparse",
        "in-context learning theory",
    ],
    "physics": [
        "quantum error correction logical qubit",
        "dark matter direct detection",
        "gravitational wave multi-messenger",
        "topological phases matter",
        "quantum gravity holography",
    ],
    "philosophy": [
        "consciousness integrated information theory critique",
        "moral patienthood artificial intelligence",
        "epistemic injustice machine learning",
        "philosophy of mind predictive processing",
    ],
    "astronomy": [
        "fast radio burst origin",
        "exoplanet atmosphere biosignature",
        "black hole merger gravitational wave",
        "fermi paradox great filter",
    ],
    "mathematics": [
        "algebraic topology data analysis",
        "random matrix theory neural networks",
        "combinatorics extremal graph theory",
    ],
    "biology": [
        "protein folding dynamics alphafold",
        "crispr base editing precision",
        "neuroscience predictive coding",
        "synthetic biology minimal cell",
    ],
}


@dataclass
class LoopState:
    """Persisted state so the loop can resume after interruption."""
    cycle: int = 0
    sessions_created: int = 0
    pages_processed: int = 0
    start_time: float = field(default_factory=time.time)
    last_cycle_time: float = 0
    last_novelty: float = 0.5
    consecutive_low_novelty: int = 0
    seen_queries: list = field(default_factory=list)
    recent_topics: list = field(default_factory=list)
    domain_novelty_history: dict = field(default_factory=dict)
    stopped: bool = False
    stop_reason: str = ""

    def elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)

    @classmethod
    def from_json(cls, s: str) -> "LoopState":
        d = json.loads(s)
        return cls(**d)


class AutonomousLoop:
    """
    KAEL's overnight learning engine.
    Runs without human supervision within defined safety limits.
    """

    def __init__(
        self,
        runner,
        store,
        counsellor=None,
        # Safety limits
        max_storage_mb: float = 500.0,
        max_hours: float = 8.0,
        max_sessions: int = 500,
        min_novelty_threshold: float = 0.25,
        max_consecutive_low_novelty: int = 5,
        cooldown_seconds: float = 10.0,
        # Behaviour
        use_counsellor_for_direction: bool = True,
        counsellor_every_n_cycles: int = 5,
        state_file: Optional[Path] = None,
    ):
        self.runner = runner
        self.store = store
        self.counsellor = counsellor

        # Safety
        self.max_storage_mb = max_storage_mb
        self.max_hours = max_hours
        self.max_sessions = max_sessions
        self.min_novelty_threshold = min_novelty_threshold
        self.max_consecutive_low_novelty = max_consecutive_low_novelty
        self.cooldown_seconds = cooldown_seconds

        # Behaviour
        self.use_counsellor = use_counsellor_for_direction and counsellor is not None
        self.counsellor_every_n_cycles = counsellor_every_n_cycles

        # State
        self.state_file = state_file or (Path("logs") / "loop_state.json")
        self.state = self._load_or_init_state()
        self._running = False
        self._crawler = None
        self._processor = None

        # Handle SIGINT / SIGTERM gracefully
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self):
        if not _check_cuda_health():
            logger.error("Aborting autonomous loop — CUDA context is broken")
            return
        """Start the autonomous loop. Runs until a stop condition is met."""
        self._running = True
        self._init_crawler()
        logger.info(self._banner())

        try:
            await self._loop()
        except Exception as e:
            logger.error(f"Loop crashed: {e}", exc_info=True)
            self.state.stop_reason = f"crash: {e}"
        finally:
            self._save_state()
            self._log_summary()

    def stop(self, reason: str = "manual"):
        self._running = False
        self.state.stopped = True
        self.state.stop_reason = reason
        logger.info(f"Loop stop requested: {reason}")

    def status(self) -> dict:
        s = self.state
        return {
            "running": self._running,
            "cycle": s.cycle,
            "sessions_created": s.sessions_created,
            "pages_processed": s.pages_processed,
            "elapsed_hours": round(s.elapsed_hours(), 2),
            "last_novelty": round(s.last_novelty, 3),
            "consecutive_low_novelty": s.consecutive_low_novelty,
            "storage_mb": round(self._get_storage_mb(), 1),
            "stop_reason": s.stop_reason,
            "seen_queries": len(s.seen_queries),
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _loop(self):
        stop_file = Path("/tmp/kael_stop")
        while self._running:
            if stop_file.exists():
                stop_file.unlink()
                self.stop("stop_file_signal")
                break
            # ── Safety checks ─────────────────────────────────────────────────
            stop = self._check_stop_conditions()
            if stop:
                self.stop(stop)
                break

            self.state.cycle += 1
            cycle_start = time.time()
            logger.info(f"\n── Cycle {self.state.cycle} ──────────────────────")
            logger.info(self._status_line())

            # ── Phase 1: Decide what to fetch ─────────────────────────────────
            query = await self._decide_next_query()
            logger.info(f"Next query: '{query}'")
            self.state.seen_queries.append(query)
            self.state.recent_topics.append(query)
            if len(self.state.recent_topics) > 200:
                self.state.recent_topics = self.state.recent_topics[-200:]

            # ── Phase 2: Fetch ────────────────────────────────────────────────
            pages = await self._fetch(query)
            if not pages:
                logger.info("No new pages this cycle")
                await asyncio.sleep(self.cooldown_seconds)
                continue

            # ── Phase 3: Process -> tau updates ───────────────────────────────
            avg_novelty = await self._process(pages)
            self.state.last_novelty = avg_novelty
            self.state.pages_processed += len(pages)

            if avg_novelty < self.min_novelty_threshold:
                self.state.consecutive_low_novelty += 1
                logger.info(f"Low novelty cycle ({avg_novelty:.3f}) — {self.state.consecutive_low_novelty}/{self.max_consecutive_low_novelty}")
            else:
                self.state.consecutive_low_novelty = 0

            # ── Phase 4: Counsellor check-in ──────────────────────────────────
            if self.use_counsellor and self.state.cycle % self.counsellor_every_n_cycles == 0:
                await self._counsellor_checkin()

            # ── Phase 5: Save state and rest ──────────────────────────────────
            self._save_state()
            if self.state.cycle % 10 == 0:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    allocated = torch.cuda.memory_allocated() / 1e6
                    logger.info(f"CUDA cache cleared | allocated: {allocated:.1f}MB")
            cycle_elapsed = time.time() - cycle_start
            rest = max(0, self.cooldown_seconds - cycle_elapsed)
            logger.info(f"Cycle {self.state.cycle} done in {cycle_elapsed:.1f}s | resting {rest:.1f}s")
            await asyncio.sleep(rest)

    # ── Curiosity engine ──────────────────────────────────────────────────────

    async def _decide_next_query(self) -> str:
        """
        Decide what to search for next.
        Priority: counsellor recommendation > curiosity templates > fallback
        """
        # Occasionally ask counsellor for direction
        if (self.use_counsellor
                and self.counsellor.available
                and self.state.cycle % self.counsellor_every_n_cycles == 0):
            tau_domains = self._get_tau_domain_weights()
            query = self.counsellor.ask_for_search_direction(
                recent_topics=self.state.recent_topics,
                tau_domains=tau_domains,
                last_novelty=self.state.last_novelty,
            )
            if query and query not in self.state.seen_queries:
                logger.info(f"Counsellor suggested: '{query}'")
                return query

        return self._curiosity_query()

    def _curiosity_query(self) -> str:
        """
        Generate a search query from curiosity templates,
        weighted toward domains with highest recent novelty contribution.
        """
        import random
        tau_domains = self._get_tau_domain_weights()

        # Weight domains by novelty — higher novelty domain gets more chances
        weighted = []
        for domain, weight in tau_domains.items():
            if domain in CURIOSITY_TEMPLATES:
                count = max(1, int(weight * 10))
                weighted.extend([domain] * count)

        if not weighted:
            weighted = list(CURIOSITY_TEMPLATES.keys())

        # Pick domain, then template, avoiding recent repeats
        random.shuffle(weighted)
        for domain in weighted:
            templates = CURIOSITY_TEMPLATES.get(domain, [])
            for template in random.sample(templates, min(len(templates), 5)):
                if template not in self.state.seen_queries[-20:]:
                    return template

        # All seen — pick oldest seen query (cycle back)
        if self.state.seen_queries:
            return self.state.seen_queries[0]

        return "persistent memory neural networks"  # absolute fallback

    def _get_tau_domain_weights(self) -> dict:
        """
        Get domain weights from session history.
        Domains with more high-importance sessions get higher weight.
        """
        from collections import defaultdict
        domain_scores = defaultdict(float)
        recent = self.store.get_recent_sessions(50)
        for s in recent:
            if s.domain and s.importance_score:
                domain_scores[s.domain] += s.importance_score

        total = sum(domain_scores.values()) or 1.0
        return {d: round(v / total, 3) for d, v in domain_scores.items()}

    # ── Fetch ─────────────────────────────────────────────────────────────────

    async def _fetch(self, query: str) -> list:
        """Fetch pages for a query. Alternates between ArXiv search and feeds."""
        try:
            if self.state.cycle % 3 == 0:
                # Every 3rd cycle: refresh feeds for new papers
                from memory.crawler import FRONTIER_FEEDS
                pages = await self._processor.crawler.crawl_feeds(
                    FRONTIER_FEEDS  # top 3 feeds to keep it fast
                )
            else:
                # Most cycles: targeted search
                pages = await self._processor.crawler.search_arxiv(
                    query=query,
                    max_results=15,
                )
            return pages
        except Exception as e:
            logger.warning(f"Fetch error: {e}")
            return []

    # ── Process ───────────────────────────────────────────────────────────────

    async def _process(self, pages: list) -> float:
        """Process pages and return average novelty."""
        if not pages:
            return 0.0

        result = await self._processor._process_pages(
            pages,
            mode="digest",       # digest mode: efficient, one session per batch
            source="autonomous",
        )
        self.state.sessions_created += result.get("sessions", 0)

        # Get average novelty from recent sessions
        recent = self.store.get_recent_sessions(result.get("sessions", 1))
        if recent:
            novelties = [s.novelty_score for s in recent if s.novelty_score is not None]
            return sum(novelties) / len(novelties) if novelties else 0.5
        return 0.5

    # ── Counsellor check-in ───────────────────────────────────────────────────

    async def _counsellor_checkin(self):
        """
        Periodic check-in with Perplexity.
        KAEL reports what it's been learning and gets direction.
        """
        if not self.counsellor or not self.counsellor.available:
            return

        recent = self.state.recent_topics[-5:]
        question = (
            f"I've completed {self.state.cycle} learning cycles. "
            f"Recent topics: {', '.join(recent)}. "
            f"Current novelty: {self.state.last_novelty:.3f}. "
            f"Sessions created: {self.state.sessions_created}. "
            f"Is there a direction I should pivot toward, or should I continue this trajectory?"
        )
        result = self.counsellor.consult(question, trigger="periodic_checkin")
        if result:
            # Feed the counsellor response back as a session
            self.runner.run(
                user_input=f"Counsel from Perplexity:\n\n{result.response}",
                capture_embedding=True,
            )
            self.state.sessions_created += 1
            logger.info(f"Counsellor check-in complete | response: {result.response[:100]}...")

    # ── Safety checks ─────────────────────────────────────────────────────────

    def _check_stop_conditions(self) -> Optional[str]:
        s = self.state

        if s.elapsed_hours() >= self.max_hours:
            return f"time limit reached ({self.max_hours}h)"

        if s.sessions_created >= self.max_sessions:
            return f"session limit reached ({self.max_sessions})"

        storage = self._get_storage_mb()
        if storage >= self.max_storage_mb:
            return f"storage limit reached ({storage:.1f}MB >= {self.max_storage_mb}MB)"

        if s.consecutive_low_novelty >= self.max_consecutive_low_novelty:
            return f"novelty exhausted ({self.max_consecutive_low_novelty} consecutive low cycles)"

        return None

    def _get_storage_mb(self) -> float:
        """Total size of DB + crawl cache in MB."""
        total = 0
        from config import cfg
        db = cfg.memory.db_path
        if db.exists():
            total += db.stat().st_size
        cache = cfg.logging.log_dir / "crawl_cache"
        if cache.exists():
            total += sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
        return total / (1024 * 1024)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _init_crawler(self):
        from memory.crawler_processor import CrawlProcessor
        self._processor = CrawlProcessor(self.runner, self.store)

    def _load_or_init_state(self) -> LoopState:
        if self.state_file.exists():
            try:
                state = LoopState.from_json(self.state_file.read_text())
                # If previous run ended, reset counters but keep seen_queries
                # so we don't re-search the same topics
                if state.stopped or state.elapsed_hours() >= self.max_hours:
                    logger.info(
                        f"Previous loop ended ({state.stop_reason}) — "
                        f"resetting clock, keeping {len(state.seen_queries)} seen queries"
                    )
                    fresh = LoopState()
                    fresh.seen_queries = state.seen_queries   # don't re-search these
                    fresh.recent_topics = state.recent_topics # continuity for counsellor
                    fresh.domain_novelty_history = state.domain_novelty_history
                    return fresh
            except Exception:
                pass
        return LoopState()

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(self.state.to_json())

    def _handle_stop(self, signum, frame):
        logger.info(f"Signal {signum} received — stopping cleanly")
        self.stop("signal")

    def _banner(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"KAEL Autonomous Loop starting\n"
            f"  max_hours={self.max_hours} | max_sessions={self.max_sessions}\n"
            f"  max_storage={self.max_storage_mb}MB | cooldown={self.cooldown_seconds}s\n"
            f"  counsellor={'active' if self.use_counsellor else 'inactive'}\n"
            f"{'='*50}"
        )

    def _status_line(self) -> str:
        s = self.state
        return (
            f"cycle={s.cycle} sessions={s.sessions_created} "
            f"pages={s.pages_processed} "
            f"novelty={s.last_novelty:.3f} "
            f"storage={self._get_storage_mb():.1f}MB "
            f"elapsed={s.elapsed_hours():.2f}h"
        )

    def _log_summary(self):
        s = self.state
        logger.info(
            f"\n{'='*50}\n"
            f"KAEL Autonomous Loop ended\n"
            f"  cycles: {s.cycle}\n"
            f"  sessions created: {s.sessions_created}\n"
            f"  pages processed: {s.pages_processed}\n"
            f"  elapsed: {s.elapsed_hours():.2f}h\n"
            f"  stop reason: {s.stop_reason}\n"
            f"  final novelty: {s.last_novelty:.3f}\n"
            f"{'='*50}"
        )


# ── Standalone runner ─────────────────────────────────────────────────────────

async def _run_standalone():
    """Entry point when run as: python -m memory.autonomous_loop"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from config import cfg
    from core.model import KAELModel
    from core.runner import SessionRunner
    from memory.session_store import SessionStore
    from core.counsellor import KAELCounsellor

    print("Loading KAEL for autonomous loop...")
    model = KAELModel(cfg)
    store = SessionStore()
    runner = SessionRunner(model, store)

    # Activate Phase 1 if not already
    if model._phase < 1:
        model.activate_phase1()
        print("Phase 1 activated")

    counsellor = KAELCounsellor()

    # Read limits from env or use defaults
    loop = AutonomousLoop(
        runner=runner,
        store=store,
        counsellor=counsellor,
        max_storage_mb=float(os.environ.get("KAEL_MAX_STORAGE_MB", "500")),
        max_hours=float(os.environ.get("KAEL_MAX_HOURS", "8")),
        max_sessions=int(os.environ.get("KAEL_MAX_SESSIONS", "500")),
        cooldown_seconds=float(os.environ.get("KAEL_COOLDOWN_SECONDS", "10")),
    )

    await loop.start()


if __name__ == "__main__":
    asyncio.run(_run_standalone())