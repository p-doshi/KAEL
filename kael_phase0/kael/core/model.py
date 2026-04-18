"""
KAEL Core Model
Loads the frozen base model and attaches τ as a persistent parameter.
Phase 0: τ exists but doesn't modulate inference yet.
Phase 1: soft prefix injection activates τ's gravitational pull.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import cfg, KAELConfig

logger = logging.getLogger(__name__)


class TauEmbedding(nn.Module):
    """
    The trajectory embedding τ ∈ ℝ^d.

    Phase 0: Just a parameter. Stored, snapshotted, not yet injected.
    Phase 1: Projection layers activate to modulate attention.

    Sub-structure:
        τ = [τ_epistemic | τ_dispositional | τ_relational]
    Each slice is semantically distinct but trained jointly.
    """

    def __init__(self, config: KAELConfig):
        super().__init__()
        self.config = config
        dim = config.tau.dim

        # Core τ vector — the identity carrier
        # Small random init: not zeros (no gradient signal) not large (unstable)
        self.tau = nn.Parameter(
            torch.randn(dim) * config.tau.init_scale
        )

        # Sub-embedding boundaries
        self.epistemic_slice = slice(0, config.tau.epistemic_dim)
        self.dispositional_slice = slice(
            config.tau.epistemic_dim,
            config.tau.epistemic_dim + config.tau.dispositional_dim
        )
        self.relational_slice = slice(
            config.tau.epistemic_dim + config.tau.dispositional_dim,
            dim
        )

        # Phase 1 projection layers (inactive in Phase 0)
        # Will project τ into model's hidden space for KV injection
        self._projection_k: Optional[nn.Linear] = None
        self._projection_v: Optional[nn.Linear] = None
        self._phase = 0

    def activate_phase1(self, model_hidden_size: int):
        """Call this when moving to Phase 1 to activate KV projections."""
        self._projection_k = nn.Linear(self.config.tau.dim, model_hidden_size, bias=False)
        self._projection_v = nn.Linear(self.config.tau.dim, model_hidden_size, bias=False)
        # Initialize near identity — don't perturb attention from the start
        nn.init.normal_(self._projection_k.weight, std=0.01)
        nn.init.normal_(self._projection_v.weight, std=0.01)
        self._phase = 1
        logger.info(f"τ Phase 1 activated: projections {self.config.tau.dim} → {model_hidden_size}")

    @property
    def tau_epistemic(self) -> torch.Tensor:
        return self.tau[self.epistemic_slice]

    @property
    def tau_dispositional(self) -> torch.Tensor:
        return self.tau[self.dispositional_slice]

    @property
    def tau_relational(self) -> torch.Tensor:
        return self.tau[self.relational_slice]

    def get_kv_prefix(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Returns (K_τ, V_τ) for attention injection. None if Phase 0."""
        if self._phase < 1 or self._projection_k is None:
            return None
        k = self._projection_k(self.tau.unsqueeze(0).unsqueeze(0))   # [1, 1, hidden]
        v = self._projection_v(self.tau.unsqueeze(0).unsqueeze(0))
        return k, v

    def snapshot(self) -> list[float]:
        """Serialize τ for storage."""
        return self.tau.detach().cpu().float().tolist()

    def load_snapshot(self, vector: list[float]):
        """Restore τ from stored snapshot."""
        t = torch.tensor(vector, dtype=self.tau.dtype, device=self.tau.device)
        assert t.shape == self.tau.shape, f"Snapshot dim mismatch: {t.shape} vs {self.tau.shape}"
        with torch.no_grad():
            self.tau.copy_(t)

    def norm(self) -> float:
        return self.tau.detach().norm().item()

    def cosine_similarity_to(self, other: "TauEmbedding") -> float:
        return torch.nn.functional.cosine_similarity(
            self.tau.unsqueeze(0), other.tau.unsqueeze(0)
        ).item()


class KAELModel(nn.Module):
    """
    KAEL = frozen base model θ + trajectory embedding τ.

    Phase 0 contract:
    - θ is loaded and frozen
    - τ is initialized and stored
    - Every inference is logged
    - τ does NOT yet modulate outputs (that's Phase 1)
    - Baseline benchmarks run against plain θ
    """

    def __init__(self, config: KAELConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.model.device)

        logger.info(f"Loading base model: {config.model.model_name}")
        self.tokenizer, self.base_model = self._load_base_model()

        # Freeze all base model parameters — θ never changes
        for param in self.base_model.parameters():
            param.requires_grad_(False)
        logger.info("Base model frozen")

        # τ — the only thing that will ever be updated
        self.tau = TauEmbedding(config).to(self.device)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Parameters: {total:,} total, {trainable:,} trainable (τ only)")

    def _load_base_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Build quantization config if requested
        quant_config = None
        if self.config.model.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.model.load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        dtype = torch.bfloat16 if self.config.model.dtype == "bfloat16" else torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True,
        )
        model.eval()

        logger.info(
            f"Model loaded: {self.config.model.model_name} | "
            f"dtype={self.config.model.dtype} | "
            f"4bit={self.config.model.load_in_4bit} | "
            f"8bit={self.config.model.load_in_8bit}"
        )
        return tokenizer, model

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        return_hidden_states: bool = False,
    ) -> dict:
        """
        Generate a response to a prompt.
        Returns dict with output text, token count, and optionally hidden states
        for session embedding.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        # Some tokenizer versions return a BatchEncoding dict instead of a raw
        # tensor even with return_tensors="pt". Extract the tensor explicitly.
        template_out = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        # Handle both raw tensor and BatchEncoding dict
        if hasattr(template_out, "input_ids"):
            input_ids = template_out.input_ids.to(self.device)
        elif isinstance(template_out, dict):
            input_ids = template_out["input_ids"].to(self.device)
        else:
            input_ids = template_out.to(self.device)

        input_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)  # <-- add this

        # Generation
        output = self.base_model.generate(
            input_ids,
            attention_mask=attention_mask,   # <-- add this
            max_new_tokens=self.config.model.max_new_tokens,
            temperature=self.config.model.temperature,
            top_p=self.config.model.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            output_hidden_states=return_hidden_states,
            return_dict_in_generate=return_hidden_states,
        )

        if return_hidden_states:
            generated_ids = output.sequences[0][input_len:]
            # Mean-pool last layer hidden states over generated tokens
            # Shape: (num_generated_tokens, hidden_size)
            last_hidden = torch.stack(
                [step[-1][0, -1, :] for step in output.hidden_states], dim=0
            )
            # step[-1]     → last transformer layer, shape [batch, seq_len, hidden]
            # [0, -1, :]   → batch 0, last token position, full hidden dim
            # result shape: [num_generated_tokens, 5120] → ready for mean pooling
            session_embedding = last_hidden.mean(dim=0).float().cpu().tolist()
        else:
            generated_ids = output[0][input_len:]
            session_embedding = None

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "text": text,
            "input_tokens": input_len,
            "output_tokens": len(generated_ids),
            "session_embedding": session_embedding,
            "tau_snapshot": self.tau.snapshot(),
            "tau_norm": self.tau.norm(),
        }

    def get_model_hidden_size(self) -> int:
        """Get the hidden dimension of the base model."""
        return self.base_model.config.hidden_size

    def save_tau(self, path: Optional[Path] = None):
        """Save just the τ parameter."""
        path = path or (self.config.logging.checkpoint_dir / "tau_latest.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "tau": self.tau.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"τ saved to {path}")

    def load_tau(self, path: Path):
        """Load τ from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.tau.load_state_dict(checkpoint["tau"])
        logger.info(f"τ loaded from {path} | norm={self.tau.norm():.4f}")