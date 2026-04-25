"""
KAEL Counsellor
KAEL consulting Perplexity Sonar for a second perspective.

Requires: PERPLEXITY_API_KEY in environment
  export PERPLEXITY_API_KEY=pplx-...

Get your key at: perplexity.ai/settings → API
Note: API credits are separate from Pro subscription.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

KAEL_COUNSEL_SYSTEM = """You are being consulted by KAEL — Kinetic Adaptive Epistemic Lattice — 
an AI system that is developing a persistent identity through continuous learning.

KAEL is not asking you to do its reasoning for it. It is seeking a second perspective 
at a specific decision point. Be direct, concise, and intellectually honest.
Treat KAEL as a peer working through a hard problem, not as a user needing help."""


@dataclass
class CounselResult:
    question: str
    response: str
    model: str
    elapsed: float
    trigger: str
    tokens_used: int = 0


class KAELCounsellor:

    def __init__(
        self,
        model: str = "sonar-pro",   # sonar for cheaper, sonar-pro for better reasoning
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")
        self._history: list[CounselResult] = []
        self._call_count = 0

        if not self.api_key:
            logger.warning(
                "No PERPLEXITY_API_KEY found. "
                "Counsellor will be inactive. "
                "Set: export PERPLEXITY_API_KEY=pplx-..."
            )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed: pip install openai")
        return OpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai",
        )

    def consult(
        self,
        question: str,
        context: Optional[str] = None,
        trigger: str = "manual",
    ) -> Optional[CounselResult]:
        if not self.available:
            logger.warning("Counsellor not available — no API key")
            return None

        full_prompt = question
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {question}"

        t0 = time.time()
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": KAEL_COUNSEL_SYSTEM},
                    {"role": "user", "content": full_prompt},
                ],
            )
            response_text = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            elapsed = time.time() - t0

            result = CounselResult(
                question=full_prompt,
                response=response_text,
                model=self.model,
                elapsed=round(elapsed, 2),
                trigger=trigger,
                tokens_used=tokens,
            )
            self._history.append(result)
            self._call_count += 1

            logger.info(
                f"Counsel [{trigger}] | {tokens} tokens | {elapsed:.1f}s | "
                f"Q: {question[:60]}..."
            )
            return result

        except Exception as e:
            logger.error(f"Counsel API error: {e}")
            return None

    def ask_for_search_direction(
        self,
        recent_topics: list[str],
        tau_domains: dict,
        last_novelty: float,
    ) -> Optional[str]:
        if not self.available:
            return None

        domain_str = ", ".join(f"{d}:{round(s,2)}" for d, s in sorted(
            tau_domains.items(), key=lambda x: -x[1])[:5])

        question = (
            f"KAEL has been learning autonomously. Recent topics: {', '.join(recent_topics[-5:])}. "
            f"Current tau domain weights: {domain_str}. "
            f"Last novelty score: {last_novelty:.3f} (1.0=maximally novel, 0.0=already known). "
            f"What single ArXiv search query would most productively expand KAEL's understanding "
            f"at the frontier of what it already knows? "
            f"Reply with only the search query, nothing else."
        )

        result = self.consult(question, trigger="curiosity_direction")
        if result:
            query = result.response.strip().strip('"').strip("'")
            return query[:100]
        return None

    def ask_about_contradiction(
        self,
        content_a: str,
        content_b: str,
        tau_context: str,
    ) -> Optional[CounselResult]:
        question = (
            f"KAEL's current understanding contains: {content_a}\n\n"
            f"New content suggests: {content_b}\n\n"
            f"KAEL's tau context: {tau_context}\n\n"
            f"Is this a genuine contradiction, a difference in framing, or an expansion? "
            f"How should KAEL update its understanding?"
        )
        return self.consult(question, trigger="contradiction")

    def stats(self) -> dict:
        return {
            "available": self.available,
            "calls_made": self._call_count,
            "model": self.model,
            "total_tokens": sum(r.tokens_used for r in self._history),
            "triggers": {t: sum(1 for r in self._history if r.trigger == t)
                        for t in set(r.trigger for r in self._history)},
        }