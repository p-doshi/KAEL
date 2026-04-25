"""
KAEL Novelty Scorer
Measures how novel a session's content is relative to τ_epistemic.

Used for:
  - Gate g_t input (novelty is one of three signals)
  - Importance scoring in M
  - Directed growth β weight (novel sessions pull toward frontier)

Phase 1 approach: cosine distance in embedding space.
  novelty = 1 - cosine_similarity(session_embedding, τ_epistemic_projected)

High novelty (→ 1.0): session content far from current τ_epistemic
Low novelty (→ 0.0): session content aligned with τ_epistemic (deepening)

Neither is "better" — the gate and consolidation network decide what to do
with novelty. High novelty + consistent = expansion. High novelty + contradictory
= flag for human loop.
"""

import torch
import torch.nn.functional as F
import logging
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class NoveltyScorer:
    """
    Scores session novelty against τ_epistemic.

    Two scoring modes:
      1. embedding_cosine: 1 - cos_sim(session_emb, τ_epistemic_projected)
         Requires session_embedding captured during generation.
         Most accurate.

      2. keyword_entropy: based on domain keyword distribution vs τ history
         Fallback when no session embedding available.
    """

    def __init__(self, tau_module, device: str = "cpu"):
        self.tau_module = tau_module
        self.device = torch.device(device)

        # Running history of session embeddings for relative novelty
        # Bounded buffer — don't grow unbounded
        self._embedding_history: list[torch.Tensor] = []
        self._max_history = 200
        self._domain_history: list[str] = []

    def score(
        self,
        session_embedding: Optional[list[float]],
        domain: Optional[str] = None,
    ) -> float:
        """
        Compute novelty score for a session. Returns float in [0, 1].
        1.0 = maximally novel, 0.0 = perfectly aligned with current τ.
        """
        if session_embedding is not None:
            score = self._embedding_novelty(session_embedding)
        else:
            score = self._keyword_novelty(domain)

        # Update history
        if session_embedding is not None:
            emb = torch.tensor(session_embedding, dtype=torch.float32, device=self.device)
            self._embedding_history.append(emb)
            if len(self._embedding_history) > self._max_history:
                self._embedding_history.pop(0)

        if domain:
            self._domain_history.append(domain)
            if len(self._domain_history) > self._max_history:
                self._domain_history.pop(0)

        return round(float(score), 4)

    def _embedding_novelty(self, session_embedding: list[float]) -> float:
        """
        1 - cosine_similarity(session_embedding, τ_epistemic)

        τ_epistemic lives in τ-space (dim d).
        session_embedding lives in model hidden space (also dim d if mean-pooled).
        They're in the same space so direct comparison is valid.
        """
        session_emb = torch.tensor(
            session_embedding, dtype=torch.float32, device=self.device
        )

        # Use τ_epistemic slice
        tau_ep = self.tau_module.tau_epistemic.detach().to(self.device).float()

        # Ensure same dim — τ_epistemic may be smaller than full hidden_size
        # Align by truncating or padding
        min_dim = min(session_emb.shape[0], tau_ep.shape[0])
        session_emb = session_emb[:min_dim]
        tau_ep = tau_ep[:min_dim]

        sim = F.cosine_similarity(
            session_emb.unsqueeze(0),
            tau_ep.unsqueeze(0),
            dim=1,
            eps=1e-8,
        ).item()

        # sim in [-1, 1] → novelty in [0, 1]
        # sim=1 (identical direction) → novelty=0 (already known)
        # sim=-1 (opposite) → novelty=1 (maximally novel/contradictory)
        # sim=0 (orthogonal) → novelty=0.5 (genuinely new territory)
        novelty = (1.0 - sim) / 2.0
        return novelty

    def _keyword_novelty(self, domain: Optional[str]) -> float:
        """
        Fallback novelty estimate based on domain distribution entropy.
        Novel = domain not seen recently.
        """
        if not domain or not self._domain_history:
            return 0.5  # Unknown — assume moderate novelty

        recent = self._domain_history[-50:]
        domain_count = recent.count(domain)
        # If domain appears in last 50, it's less novel
        novelty = 1.0 - (domain_count / len(recent))
        return novelty

    def relative_novelty(self, session_embedding: list[float]) -> float:
        """
        Novelty relative to recent session history (not just τ).
        Useful for detecting when a genuinely new topic appears
        even if τ hasn't seen it before.
        """
        if not self._embedding_history:
            return 0.5

        session_emb = torch.tensor(
            session_embedding, dtype=torch.float32, device=self.device
        )
        history_stack = torch.stack(self._embedding_history[-20:])

        # Mean similarity to recent sessions
        sims = F.cosine_similarity(
            session_emb.unsqueeze(0).expand(history_stack.shape[0], -1),
            history_stack,
            dim=1,
            eps=1e-8,
        )
        mean_sim = sims.mean().item()
        return (1.0 - mean_sim) / 2.0

    def combined_novelty(
        self,
        session_embedding: Optional[list[float]],
        domain: Optional[str],
        tau_weight: float = 0.7,
        history_weight: float = 0.3,
    ) -> float:
        """
        Weighted combination of τ-relative and history-relative novelty.
        Default: 70% vs τ, 30% vs recent history.
        """
        tau_novelty = self._embedding_novelty(session_embedding) if session_embedding else 0.5
        hist_novelty = self.relative_novelty(session_embedding) if session_embedding else self._keyword_novelty(domain)
        return tau_weight * tau_novelty + history_weight * hist_novelty