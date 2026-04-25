"""
KAEL τ Updater — Phase 1
Gated integration: τ_{t+1} = (1 − g_t) · τ_t + g_t · Δτ_t

Phase 1 uses a simplified update rule — no trained consolidation network yet.
Δτ is computed directly from the session embedding via a learned projection.
The gate g_t is a heuristic in Phase 1; Phase 2 trains it properly.

Phase 1 gate heuristic:
  g_t = σ(α · novelty + β · coherence_sim - γ · contradiction_signal)
  where:
    novelty       = NoveltyScorer output
    coherence_sim = cosine_sim(session_emb, τ) — how aligned is this session
    contradiction_signal = 0 (Phase 1 — no contradiction detection yet)

This gives sensible behavior without training:
  - Novel + aligned session (deepens knowledge): g_t moderate-high → partial integrate
  - Novel + orthogonal (new territory): g_t moderate → expand carefully
  - Familiar (routine): g_t low → mostly preserve τ

Directed growth:
  Δτ = α · coherence_gradient + β · novelty_gradient
  coherence_gradient = τ normalized (pull toward self-consistency)
  novelty_gradient   = session_embedding projected to τ-space (pull toward new content)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    gate_value: float
    novelty_score: float
    delta_tau_norm: float
    tau_norm_before: float
    tau_norm_after: float
    cosine_drift: float        # cosine similarity between old and new τ (1.0 = no change)
    update_applied: bool
    flagged_for_review: bool   # g_t below contradiction_threshold


class TauUpdater(nn.Module):
    """
    Computes and applies τ updates after each session.

    Learnable components (small, trainable):
      - session_projector: projects session_embedding into τ-space
        This is the only trained component in Phase 1.

    In Phase 3 this gets replaced by the full consolidation network f.
    """

    def __init__(self, tau_module, config, device: str = "cpu"):
        super().__init__()
        self.tau_module = tau_module
        self.config = config
        self.device = torch.device(device)

        tau_dim = config.tau.dim

        # Project session embedding (hidden_size) → τ-space (tau_dim)
        # These may differ for 3B: hidden_size=2048, tau_dim=2048 (same for 3B)
        # For other models set appropriately
        hidden_size = getattr(config.model, "_hidden_size_cache", tau_dim)
        self.session_projector = nn.Linear(hidden_size, tau_dim, bias=False).to(self.device)
        nn.init.normal_(self.session_projector.weight, std=0.001)  # Near-zero init

        # Directed growth weights (learnable scalars)
        self.alpha_coherence = nn.Parameter(torch.tensor(config.consolidation.alpha_coherence))
        self.beta_novelty = nn.Parameter(torch.tensor(config.consolidation.beta_novelty))

        # Heuristic gate weights (fixed in Phase 1, trained in Phase 2)
        self._gate_novelty_weight = 0.4
        self._gate_coherence_weight = 0.4
        self._gate_base = 0.2   # Minimum gate value (always integrate a little)

        self.contradiction_threshold = config.gate.contradiction_threshold

    def set_hidden_size(self, hidden_size: int):
        """Call after model loads to fix projection dimension."""
        tau_dim = self.config.tau.dim
        if self.session_projector.in_features != hidden_size:
            self.session_projector = nn.Linear(hidden_size, tau_dim, bias=False).to(self.device)
            nn.init.normal_(self.session_projector.weight, std=0.001)
            logger.info(f"Session projector resized: {hidden_size} → {tau_dim}")

    def compute_gate(
        self,
        session_embedding: torch.Tensor,
        novelty_score: float,
    ) -> float:
        """
        Phase 1 heuristic gate.
        g_t ∈ [0, 1]: how much to integrate this session into τ.
        """
        # Coherence: how aligned is session with current τ?
        tau_vec = self.tau_module.tau.detach().to(self.device).float()
        min_dim = min(session_embedding.shape[0], tau_vec.shape[0])

        coherence = F.cosine_similarity(
            session_embedding[:min_dim].unsqueeze(0),
            tau_vec[:min_dim].unsqueeze(0),
            dim=1,
            eps=1e-8,
        ).item()
        # coherence in [-1, 1] → normalize to [0, 1]
        coherence_norm = (coherence + 1.0) / 2.0

        # Gate: weighted combination
        # High novelty + moderate coherence = good candidate for expansion
        # Low novelty + high coherence = routine, preserve τ
        # High novelty + low coherence = either new territory or contradiction
        gate = (
            self._gate_base
            + self._gate_novelty_weight * novelty_score
            + self._gate_coherence_weight * coherence_norm
        )
        gate = max(0.0, min(1.0, gate))  # Clamp to [0, 1]
        return gate

    def compute_delta_tau(self, session_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute Δτ from session embedding.
        Δτ = α · coherence_gradient + β · novelty_gradient

        coherence_gradient: direction that reinforces current τ structure
        novelty_gradient:   direction of session content in τ-space
        """
        # Project session into τ-space
        session_in_tau_space = self.session_projector(
            session_embedding.float().to(self.device)
        )  # [tau_dim]

        tau_vec = self.tau_module.tau.detach().to(self.device).float()

        # coherence_gradient = normalized τ (pull toward self-consistency)
        coherence_grad = F.normalize(tau_vec, dim=0)

        # novelty_gradient = normalized session projection (pull toward new content)
        novelty_grad = F.normalize(session_in_tau_space, dim=0)

        # Weighted combination
        alpha = torch.clamp(self.alpha_coherence, 0.0, 1.0)
        beta = torch.clamp(self.beta_novelty, 0.0, 1.0)

        delta_tau = alpha * coherence_grad + beta * novelty_grad
        return delta_tau

    @torch.no_grad()
    def update(
        self,
        session_embedding: Optional[list[float]],
        novelty_score: float,
        force_gate: Optional[float] = None,
    ) -> UpdateResult:
        """
        Compute and apply gated τ update.

        Args:
            session_embedding: mean-pooled hidden state from session
            novelty_score: from NoveltyScorer
            force_gate: override gate value (used when human feedback provides decision)

        Returns:
            UpdateResult with all diagnostics
        """
        if session_embedding is None:
            logger.warning("No session embedding — skipping τ update")
            return UpdateResult(
                gate_value=0.0, novelty_score=novelty_score,
                delta_tau_norm=0.0, tau_norm_before=self.tau_module.norm(),
                tau_norm_after=self.tau_module.norm(), cosine_drift=1.0,
                update_applied=False, flagged_for_review=False,
            )

        session_emb = torch.tensor(
            session_embedding, dtype=torch.float32, device=self.device
        )

        # Compute gate
        g_t = force_gate if force_gate is not None else self.compute_gate(session_emb, novelty_score)

        # Compute Δτ
        delta_tau = self.compute_delta_tau(session_emb)

        # Gated integration: τ_{t+1} = (1 − g_t) · τ_t + g_t · Δτ_t
        tau_before = self.tau_module.tau.detach().clone()
        tau_norm_before = tau_before.norm().item()

        new_tau = (1.0 - g_t) * self.tau_module.tau.data.to(self.device) + g_t * delta_tau

        # Update in-place
        self.tau_module.tau.data.copy_(new_tau.to(self.tau_module.tau.device))

        tau_norm_after = self.tau_module.tau.detach().norm().item()

        # Cosine drift: how much did τ change direction?
        cosine_drift = F.cosine_similarity(
            tau_before.flatten().to(self.device).unsqueeze(0),
            self.tau_module.tau.detach().flatten().to(self.device).unsqueeze(0),
            dim=1, eps=1e-8,
        ).item()

        flagged = g_t < self.contradiction_threshold

        result = UpdateResult(
            gate_value=round(g_t, 4),
            novelty_score=round(novelty_score, 4),
            delta_tau_norm=round(delta_tau.norm().item(), 4),
            tau_norm_before=round(tau_norm_before, 4),
            tau_norm_after=round(tau_norm_after, 4),
            cosine_drift=round(cosine_drift, 6),
            update_applied=True,
            flagged_for_review=flagged,
        )

        logger.info(
            f"τ update | g_t={result.gate_value:.3f} | "
            f"novelty={result.novelty_score:.3f} | "
            f"drift={result.cosine_drift:.6f} | "
            f"norm {result.tau_norm_before:.4f}→{result.tau_norm_after:.4f}"
            + (" | ⚑ FLAGGED" if flagged else "")
        )

        return result