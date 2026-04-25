"""
KAEL Core Model
Loads the frozen base model and attaches tau as a persistent parameter.
Phase 0: tau exists but does not modulate inference yet.
Phase 1: soft prefix injection activates tau gravitational pull.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

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

_Phase1Injector = None

def _get_injector_class():
    global _Phase1Injector
    if _Phase1Injector is None:
        from core.attention_injection import Phase1Injector
        _Phase1Injector = Phase1Injector
    return _Phase1Injector


class TauEmbedding(nn.Module):
    """
    The trajectory embedding tau in R^d.

    Phase 0: Just a parameter. Stored, snapshotted, not yet injected.
    Phase 1: Projection layers activate to modulate attention.

    Sub-structure:
        tau = [tau_epistemic | tau_dispositional | tau_relational]
    Each slice is semantically distinct but trained jointly.
    """

    def __init__(self, config: KAELConfig):
        super().__init__()
        self.config = config
        dim = config.tau.dim

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
        self._projection_k: Optional[nn.Linear] = None
        self._projection_v: Optional[nn.Linear] = None
        self._phase = 0

    def activate_phase1(self, model_hidden_size: int):
        """Call this when moving to Phase 1 to activate KV projections."""
        self._projection_k = nn.Linear(self.config.tau.dim, model_hidden_size, bias=False)
        self._projection_v = nn.Linear(self.config.tau.dim, model_hidden_size, bias=False)
        nn.init.normal_(self._projection_k.weight, std=0.01)
        nn.init.normal_(self._projection_v.weight, std=0.01)
        device = self.tau.device
        self._projection_k = self._projection_k.to(device)
        self._projection_v = self._projection_v.to(device)
        self._phase = 1
        logger.info(f"tau Phase 1 activated: projections {self.config.tau.dim} -> {model_hidden_size} on {device}")

    @property
    def tau_epistemic(self) -> torch.Tensor:
        return self.tau[self.epistemic_slice]

    @property
    def tau_dispositional(self) -> torch.Tensor:
        return self.tau[self.dispositional_slice]

    @property
    def tau_relational(self) -> torch.Tensor:
        return self.tau[self.relational_slice]

    def get_kv_prefix(self, target_device: Optional[torch.device] = None) -> Optional[tuple]:
        """Returns (K_tau, V_tau) for attention injection. None if Phase 0."""
        if self._phase < 1 or self._projection_k is None:
            return None
        k = self._projection_k(self.tau.unsqueeze(0).unsqueeze(0))
        v = self._projection_v(self.tau.unsqueeze(0).unsqueeze(0))
        if target_device is not None:
            k = k.to(target_device)
            v = v.to(target_device)
        return k, v

    def snapshot(self) -> list:
        return self.tau.detach().cpu().float().tolist()

    def load_snapshot(self, vector: list):
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
    KAEL = frozen base model theta + trajectory embedding tau.
    """

    def __init__(self, config: KAELConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading base model: {config.model.model_name}")
        self.tokenizer, self.base_model = self._load_base_model()

        for param in self.base_model.parameters():
            param.requires_grad_(False)
        logger.info("Base model frozen")

        self.tau = TauEmbedding(config).to(self.device)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Parameters: {total:,} total, {trainable:,} trainable (tau only)")

        # Phase 1 injector — None until activate_phase1() called
        self._injector = None
        self._phase = 0

    def activate_phase1(self, alpha=None):
        """
        Activate Phase 1: hook tau into attention layers.
        Call once after loading the model.
        alpha: gravitational pull strength (default from config)
        """
        hidden_size = self.get_model_hidden_size()
        self.tau.activate_phase1(hidden_size)

        InjectorClass = _get_injector_class()
        self._injector = InjectorClass(
            base_model=self.base_model,
            tau_module=self.tau,
            alpha=alpha or self.config.tau.attention_alpha,
            layers=self.config.tau.prefix_layers,
        )
        self._injector.activate()
        self._phase = 1
        logger.info(f"Phase 1 active | hidden_size={hidden_size} | alpha={alpha or self.config.tau.attention_alpha}")

    def set_alpha(self, alpha: float):
        if self._injector:
            self._injector.set_alpha(alpha)

    def injection_stats(self) -> dict:
        if self._injector:
            return self._injector.stats()
        return {"active": False}

    def _load_base_model(self):
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # apply_chat_template returns BatchEncoding on some versions — extract tensor explicitly
        template_out = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if hasattr(template_out, "input_ids"):
            input_ids = template_out.input_ids.to(self.device)
        elif isinstance(template_out, dict):
            input_ids = template_out["input_ids"].to(self.device)
        else:
            input_ids = template_out.to(self.device)

        input_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        output = self.base_model.generate(
            input_ids,
            attention_mask=attention_mask,
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
            # step[-1] -> last transformer layer [batch, seq_len, hidden]
            # [0, -1, :] -> batch 0, last token position, full hidden dim
            last_hidden = torch.stack(
                [step[-1][0, -1, :] for step in output.hidden_states], dim=0
            )
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
        return self.base_model.config.hidden_size

    def save_tau(self, path: Optional[Path] = None):
        path = path or (self.config.logging.checkpoint_dir / "tau_latest.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "tau": self.tau.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"tau saved to {path}")

    def load_tau(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.tau.load_state_dict(checkpoint["tau"])
        logger.info(f"tau loaded from {path} | norm={self.tau.norm():.4f}")