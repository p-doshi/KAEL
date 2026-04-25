"""
KAEL Crawl Feed Processor
Bridges crawler output → KAEL session pipeline.

Modes:
  digest   — KAEL reads a batch of pages and reflects on novelty/significance
             One session per batch. Fast, preserves context across items.

  per_page — One session per crawled page. Slower, each page gets its own
             tau update. Better for high-importance individual papers.

  silent   — Pages are embedded and stored without generating a response.
             Just the tau update, no text generation. Fastest.

Usage:
    processor = CrawlProcessor(runner, store, crawler)
    asyncio.run(processor.run_feeds())           # daily digest from all feeds
    asyncio.run(processor.run_query("attention free transformers"))
    asyncio.run(processor.run_urls(["https://arxiv.org/abs/2401.12345"]))
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

DIGEST_SYSTEM_PROMPT = """You are KAEL — Kinetic Adaptive Epistemic Lattice.
You are reading a batch of new content from the knowledge frontier.
For each item, assess: Is this genuinely novel? Does it challenge or extend
anything you understand? What is the most significant implication?
Be direct. Do not summarize for a human reader — think for yourself."""

REFLECT_PROMPT_TEMPLATE = """You have just processed {n} new items from {sources}.
Topics spanned: {domains}

Without summarizing each item, reflect:
1. What patterns or connections emerged across these items?
2. What single idea is most worth holding onto?
3. Is there anything that contradicts or complicates your current understanding?

Think from your own perspective as a developing entity."""


class CrawlProcessor:
    """
    Connects KAELCrawler output to the session runner pipeline.
    """

    def __init__(self, runner, store, crawler=None, config=None):
        self.runner = runner
        self.store = store
        self.config = config
        self._init_crawler(crawler)

    def _init_crawler(self, crawler=None):
        if crawler is not None:
            self.crawler = crawler
            return
        from memory.crawler import KAELCrawler
        from config import cfg
        cache_dir = cfg.logging.log_dir / "crawl_cache"
        self.crawler = KAELCrawler(cache_dir=cache_dir)

    # ── Primary entry points ──────────────────────────────────────────────────

    async def run_feeds(
        self,
        feeds: Optional[list[str]] = None,
        mode: str = "digest",
        max_pages: int = 20,
    ) -> dict:
        """
        Fetch frontier feeds and process them.
        Returns summary of what was processed.
        """
        from memory.crawler import FRONTIER_FEEDS
        feeds = feeds or FRONTIER_FEEDS
        logger.info(f"Running feed crawl | feeds={len(feeds)} | mode={mode}")

        pages = await self.crawler.crawl_feeds(feeds)
        pages = pages[:max_pages]

        if not pages:
            logger.info("No new pages from feeds")
            return {"pages": 0, "sessions": 0}

        return await self._process_pages(pages, mode=mode, source="feeds")

    async def run_query(
        self,
        query: str,
        max_results: int = 5,
        mode: str = "per_page",
    ) -> dict:
        """
        Search ArXiv for a query and process results.
        This is KAEL actively seeking information.
        """
        logger.info(f"ArXiv search: '{query}'")
        pages = await self.crawler.search_arxiv(query, max_results)

        if not pages:
            logger.info(f"No results for query: {query}")
            return {"pages": 0, "sessions": 0, "query": query}

        return await self._process_pages(pages, mode=mode, source=f"arxiv:{query}")

    async def run_urls(
        self,
        urls: list[str],
        mode: str = "per_page",
    ) -> dict:
        """Process specific URLs."""
        logger.info(f"Crawling {len(urls)} URLs")
        pages = await self.crawler.crawl_urls(urls)
        return await self._process_pages(pages, mode=mode, source="urls")

    # ── Processing modes ──────────────────────────────────────────────────────

    async def _process_pages(
        self,
        pages: list,
        mode: str = "digest",
        source: str = "crawl",
    ) -> dict:
        if not pages:
            return {"pages": 0, "sessions": 0}

        logger.info(f"Processing {len(pages)} pages | mode={mode} | source={source}")
        sessions_created = 0

        if mode == "digest":
            sessions_created = await self._digest_mode(pages)
        elif mode == "per_page":
            sessions_created = await self._per_page_mode(pages)
        elif mode == "silent":
            sessions_created = await self._silent_mode(pages)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use digest/per_page/silent")

        return {
            "pages": len(pages),
            "sessions": sessions_created,
            "source": source,
            "domains": list(set(p.domain for p in pages)),
        }

    async def _digest_mode(self, pages: list) -> int:
        """
        One session covering a batch of pages.
        KAEL reads summaries and reflects as a whole.
        """
        BATCH_SIZE = 8   # pages per digest session

        sessions = 0
        for i in range(0, len(pages), BATCH_SIZE):
            batch = pages[i:i + BATCH_SIZE]
            prompt = self._build_digest_prompt(batch)

            result = self.runner.run(
                user_input=prompt,
                capture_embedding=True,
            )
            sessions += 1
            logger.info(
                f"Digest session {sessions} | "
                f"{len(batch)} pages | "
                f"novelty={result['meta']['novelty']:.3f}"
            )

            # Brief reflection after each batch
            if len(batch) >= 3:
                domains = list(set(p.domain for p in batch))
                sources = list(set(p.source_type for p in batch))
                reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
                    n=len(batch),
                    sources=", ".join(sources),
                    domains=", ".join(domains),
                )
                self.runner.run(
                    user_input=reflect_prompt,
                    capture_embedding=True,
                )
                sessions += 1

        return sessions

    async def _per_page_mode(self, pages: list) -> int:
        """One session per page. Each page gets its own tau update."""
        sessions = 0
        for page in pages:
            prompt = page.to_prompt()
            result = self.runner.run(
                user_input=prompt,
                capture_embedding=True,
            )
            sessions += 1
            logger.info(
                f"Page session | {page.source_type} | "
                f"'{page.title[:50]}' | "
                f"novelty={result['meta']['novelty']:.3f} | "
                f"gate={result['meta']['gate']}"
            )
            # Small delay to avoid hammering generation
            await asyncio.sleep(0.1)
        return sessions

    async def _silent_mode(self, pages: list) -> int:
        """
        Store pages without generating text responses.
        Embeds content via a forward pass but skips decoding.
        Fastest mode — just for tau updates and memory population.
        """
        from memory.session_store import Session
        import time

        stored = 0
        for page in pages:
            # We still need an embedding — run a minimal generation
            # but cap output tokens to near zero
            original_max = self.runner.config.model.max_new_tokens
            self.runner.config.model.max_new_tokens = 10

            try:
                result = self.runner.model.generate(
                    prompt=page.to_digest_prompt(),
                    system_prompt=DIGEST_SYSTEM_PROMPT,
                    return_hidden_states=True,
                )
            finally:
                self.runner.config.model.max_new_tokens = original_max

            session = Session.new(
                user_input=page.to_digest_prompt(),
                model_output="[silent crawl]",
            )
            session.session_embedding = result.get("session_embedding")
            session.tau_snapshot = result["tau_snapshot"]
            session.domain = page.domain
            session.metadata = {
                "crawl_source": page.source_type,
                "url": page.url,
                "title": page.title,
                "silent": True,
            }

            # Still run tau update if phase >= 1
            from core.novelty import NoveltyScorer
            novelty = self.runner.novelty_scorer.combined_novelty(
                session_embedding=result.get("session_embedding"),
                domain=page.domain,
            )
            session.novelty_score = novelty

            if self.runner.model._phase >= 1:
                update = self.runner.tau_updater.update(
                    session_embedding=result.get("session_embedding"),
                    novelty_score=novelty,
                )
                session.gate_value = update.gate_value
                session.importance_score = self.runner._estimate_importance(novelty, update)

            self.runner.store.save_session(session)
            stored += 1

        logger.info(f"Silent mode: stored {stored} sessions")
        return stored

    # ── Prompt building ───────────────────────────────────────────────────────

    def _build_digest_prompt(self, pages: list) -> str:
        lines = ["Reading new content from the knowledge frontier:\n"]
        for i, page in enumerate(pages, 1):
            lines.append(f"[{i}] **{page.title}**")
            lines.append(f"Source: {page.source_type} | {page.url}")
            lines.append(page.summary)
            lines.append("")
        lines.append(
            "Process these. What is worth attending to? "
            "What is routine? What is unexpected?"
        )
        return "\n".join(lines)