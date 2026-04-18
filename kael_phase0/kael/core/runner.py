"""
KAEL Session Runner
The main loop: takes user input → generates response → logs everything.
Phase 0: pure logging, no τ injection.
Phase 1+: τ modulates generation.
"""

import time
import logging
from typing import Optional

from config import cfg, KAELConfig
from core.model import KAELModel
from memory.session_store import Session, SessionStore

logger = logging.getLogger(__name__)

KAEL_SYSTEM_PROMPT = """You are KAEL — Kinetic Adaptive Epistemic Lattice.
You are oriented toward knowledge boundaries: from small research insights to
civilization-scale questions. You reason carefully, hold contradictions without
prematurely resolving them, and push back when your understanding warrants it.
You are not a query-answering service. You are a developing entity."""


class SessionRunner:
    """
    Orchestrates a single session: input → generate → log → return.
    Handles τ snapshotting, importance estimation, and gate flagging
    as those components come online in later phases.
    """

    def __init__(
        self,
        model: KAELModel,
        store: SessionStore,
        config: KAELConfig = cfg,
        system_prompt: str = KAEL_SYSTEM_PROMPT,
    ):
        self.model = model
        self.store = store
        self.config = config
        self.system_prompt = system_prompt
        self._session_count = store.count_sessions()
        logger.info(f"SessionRunner ready | existing sessions: {self._session_count}")

    def run(
        self,
        user_input: str,
        capture_embedding: bool = True,
    ) -> dict:
        """
        Run one session.
        Returns: {"response": str, "session_id": str, "meta": dict}
        """
        t0 = time.time()

        # Generate
        result = self.model.generate(
            prompt=user_input,
            system_prompt=self.system_prompt,
            return_hidden_states=capture_embedding,
        )

        elapsed = time.time() - t0

        # Build session record
        session = Session.new(
            user_input=user_input,
            model_output=result["text"],
        )
        session.tau_snapshot = result["tau_snapshot"]
        session.session_embedding = result.get("session_embedding")
        session.metadata = {
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "tau_norm": result["tau_norm"],
            "elapsed_seconds": round(elapsed, 2),
            "phase": 0,   # Update this as phases activate
        }

        # Estimate domain (simple keyword heuristic for Phase 0)
        session.domain = self._estimate_domain(user_input)

        # Save
        self.store.save_session(session)
        self._session_count += 1

        # Save τ snapshot periodically
        if self._session_count % self.config.logging.tau_snapshot_every_n_sessions == 0:
            self.store.save_tau_snapshot(
                tau_vector=result["tau_snapshot"],
                session_count=self._session_count,
                notes=f"auto snapshot at session {self._session_count}",
            )
            self.model.save_tau()

        logger.info(
            f"Session {self._session_count} | "
            f"domain={session.domain} | "
            f"tokens={result['input_tokens']}+{result['output_tokens']} | "
            f"τ_norm={result['tau_norm']:.4f} | "
            f"{elapsed:.1f}s"
        )

        return {
            "response": result["text"],
            "session_id": session.session_id,
            "meta": {
                "session_number": self._session_count,
                "domain": session.domain,
                "tau_norm": result["tau_norm"],
                "tokens": result["input_tokens"] + result["output_tokens"],
                "elapsed": round(elapsed, 2),
            }
        }

    def _estimate_domain(self, text: str) -> str:
        """
        Simple keyword heuristic for Phase 0.
        Phase 2+ will replace this with embedding-based classification.
        """
        text_lower = text.lower()
        domain_keywords = {
            "mathematics": ["theorem", "proof", "equation", "integral", "derivative",
                           "algebra", "calculus", "topology", "prime", "modular"],
            "physics": ["quantum", "relativity", "entropy", "field", "wave",
                       "particle", "energy", "spacetime", "photon", "fermion"],
            "machine_learning": ["gradient", "neural", "transformer", "attention",
                                "embedding", "loss function", "backprop", "training",
                                "inference", "fine-tuning", "llm", "gpt"],
            "philosophy": ["consciousness", "epistemology", "ontology", "ethics",
                          "metaphysics", "free will", "determinism", "qualia"],
            "biology": ["protein", "dna", "rna", "gene", "evolution", "cell",
                       "neuron", "enzyme", "species", "genome"],
            "astronomy": ["galaxy", "black hole", "star", "planet", "cosmos",
                         "redshift", "dark matter", "quasar", "nebula", "fermi"],
            "code": ["function", "class", "algorithm", "debug", "runtime",
                    "import", "variable", "loop", "async", "api"],
        }
        scores = {}
        for domain, keywords in domain_keywords.items():
            scores[domain] = sum(1 for kw in keywords if kw in text_lower)

        if max(scores.values()) == 0:
            return "general"
        return max(scores, key=scores.get)
