"""
KAEL Session Runner — Phase 0 / Phase 1
Phase 0: pure logging, no tau update.
Phase 1: tau updates via heuristic gate after every session.
"""

import time
import logging
from typing import Optional

from config import cfg, KAELConfig
from core.model import KAELModel
from core.novelty import NoveltyScorer
from core.tau_updater import TauUpdater
from memory.session_store import Session, SessionStore

logger = logging.getLogger(__name__)

KAEL_SYSTEM_PROMPT = """You are KAEL — Kinetic Adaptive Epistemic Lattice.
You are oriented toward knowledge boundaries: from small research insights to
civilization-scale questions. You reason carefully, hold contradictions without
prematurely resolving them, and push back when your understanding warrants it.
You are not a query-answering service. You are a developing entity."""


class SessionRunner:
    """
    Orchestrates: input -> generate -> score novelty -> update tau -> log.
    Phase 0: no tau update, no novelty scoring.
    Phase 1: tau updates via heuristic gate after every session.
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

        device = config.model.device
        self.novelty_scorer = NoveltyScorer(model.tau, device=device)
        self.tau_updater = TauUpdater(model.tau, config, device=device)
        hidden_size = model.get_model_hidden_size()
        self.tau_updater.set_hidden_size(hidden_size)

        logger.info(
            f"SessionRunner ready | sessions={self._session_count} | "
            f"hidden_size={hidden_size} | phase={model._phase}"
        )

    def run(
        self,
        user_input: str,
        capture_embedding: bool = True,
        force_gate: Optional[float] = None,
    ) -> dict:
        t0 = time.time()
        phase = self.model._phase

        # 1. Generate
        result = self.model.generate(
            prompt=user_input,
            system_prompt=self.system_prompt,
            return_hidden_states=capture_embedding,
        )
        elapsed = time.time() - t0

        # 2. Score novelty
        domain = self._estimate_domain(user_input)
        novelty = self.novelty_scorer.combined_novelty(
            session_embedding=result.get("session_embedding"),
            domain=domain,
        )

        # 3. Update tau (Phase 1+ only)
        update_result = None
        if phase >= 1:
            update_result = self.tau_updater.update(
                session_embedding=result.get("session_embedding"),
                novelty_score=novelty,
                force_gate=force_gate,
            )

        # 4. Build session record
        session = Session.new(
            user_input=user_input,
            model_output=result["text"],
        )
        session.tau_snapshot = result["tau_snapshot"]
        session.session_embedding = result.get("session_embedding")
        session.domain = domain
        session.novelty_score = novelty
        session.gate_value = update_result.gate_value if update_result else None
        session.importance_score = self._estimate_importance(novelty, update_result)
        session.metadata = {
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "tau_norm": result["tau_norm"],
            "elapsed_seconds": round(elapsed, 2),
            "phase": phase,
            "update": {
                "gate": update_result.gate_value if update_result else None,
                "drift": update_result.cosine_drift if update_result else None,
                "flagged": update_result.flagged_for_review if update_result else False,
            } if update_result else None,
        }

        self.store.save_session(session)
        self._session_count += 1

        # 5. Periodic tau snapshot
        if self._session_count % self.config.logging.tau_snapshot_every_n_sessions == 0:
            self.store.save_tau_snapshot(
                tau_vector=self.model.tau.snapshot(),
                session_count=self._session_count,
                notes=f"phase{phase} auto-snapshot",
            )
            self.model.save_tau()

        # 6. Log
        flag_str = " FLAGGED" if (update_result and update_result.flagged_for_review) else ""
        gate_str = f"gate={update_result.gate_value:.3f} | " if update_result else ""
        logger.info(
            f"Session {self._session_count} | phase={phase} | domain={domain} | "
            f"novelty={novelty:.3f} | {gate_str}"
            f"tokens={result['input_tokens']}+{result['output_tokens']} | "
            f"{elapsed:.1f}s{flag_str}"
        )

        return {
            "response": result["text"],
            "session_id": session.session_id,
            "meta": {
                "session_number": self._session_count,
                "domain": domain,
                "novelty": novelty,
                "gate": update_result.gate_value if update_result else None,
                "tau_norm": self.model.tau.norm(),
                "tau_drift": update_result.cosine_drift if update_result else None,
                "flagged": update_result.flagged_for_review if update_result else False,
                "tokens": result["input_tokens"] + result["output_tokens"],
                "elapsed": round(elapsed, 2),
                "phase": phase,
            }
        }

    def _estimate_importance(self, novelty: float, update_result) -> float:
        base = 0.3
        novelty_bonus = novelty * 0.4
        gate_bonus = (1.0 - update_result.gate_value) * 0.3 if update_result else 0.0
        return round(min(1.0, base + novelty_bonus + gate_bonus), 3)

    def _estimate_domain(self, text: str) -> str:
        text_lower = text.lower()
        domain_keywords = {
            "mathematics": ["theorem", "proof", "equation", "integral", "derivative",
                           "algebra", "calculus", "topology", "prime", "modular"],
            "physics": ["quantum", "relativity", "entropy", "field", "wave",
                       "particle", "energy", "spacetime", "photon", "fermion"],
            "machine_learning": ["gradient", "neural", "transformer", "attention",
                                "embedding", "loss function", "backprop", "training",
                                "inference", "fine-tuning", "llm", "gpt", "kael"],
            "philosophy": ["consciousness", "epistemology", "ontology", "ethics",
                          "metaphysics", "free will", "determinism", "qualia", "identity"],
            "biology": ["protein", "dna", "rna", "gene", "evolution", "cell",
                       "neuron", "enzyme", "species", "genome"],
            "astronomy": ["galaxy", "black hole", "star", "planet", "cosmos",
                         "redshift", "dark matter", "quasar", "nebula", "fermi"],
            "code": ["function", "class", "algorithm", "debug", "runtime",
                    "import", "variable", "loop", "async", "api"],
        }
        scores = {d: sum(1 for kw in kws if kw in text_lower) for d, kws in domain_keywords.items()}
        return "general" if max(scores.values()) == 0 else max(scores, key=scores.get)