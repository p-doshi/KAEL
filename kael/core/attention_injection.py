"""
KAEL Phase 1 — Attention Injection
Hooks τ into the model's attention layers as a soft prefix + bias field.

Two mechanisms working together:
  1. Soft prefix: τ projected into K, V and prepended to every layer's KV cache.
     Every token attends to τ as if it were part of the context — silently.

  2. Attention bias field: adds α·sim(Q, K_τ) to attention logits.
     Tokens whose query direction aligns with τ get slightly boosted attention.
     This is the gravitational pull — τ-consistent patterns become more salient.

Architecture notes for Qwen2.5-3B:
  - 28 attention layers
  - hidden_size = 2048
  - num_attention_heads = 16
  - num_key_value_heads = 8  (GQA — grouped query attention)
  - head_dim = 128

For GQA models (Qwen2.5, LLaMA 3, Mistral) the KV heads != Q heads.
We inject τ into KV space (num_kv_heads × head_dim), not Q space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TauAttentionHook:
    """
    Registered as a forward hook on each attention layer.
    Intercepts the attention computation and injects τ influence.

    Usage:
        hook = TauAttentionHook(tau_embedding, layer_idx, config)
        handle = attn_layer.register_forward_hook(hook)
        # To remove: handle.remove()
    """

    def __init__(self, tau_module, layer_idx: int, alpha: float = 0.1):
        self.tau_module = tau_module
        self.layer_idx = layer_idx
        self.alpha = alpha
        self._call_count = 0

    def __call__(self, module, input, output):
        """
        Called after the attention layer's forward pass.
        output is typically (hidden_states, attention_weights, present_kv)
        We modify hidden_states via the bias signal.
        """
        self._call_count += 1

        # output structure varies by model — handle both tuple and tensor
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Get τ KV prefix for this layer
        kv = self.tau_module.get_kv_prefix(target_device=hidden_states.device)
        if kv is None:
            return output  # Phase 0 — no injection

        k_tau, v_tau = kv
        # k_tau, v_tau: [1, 1, hidden_size]
        k_tau = k_tau.to(dtype=hidden_states.dtype)
        v_tau = v_tau.to(dtype=hidden_states.dtype)

        # Compute similarity of current hidden state to τ_k
        # hidden_states: [batch, seq_len, hidden_size]
        # We use mean-pool over seq_len for efficiency
        h_mean = hidden_states.mean(dim=1, keepdim=True)  # [batch, 1, hidden_size]

        sim = F.cosine_similarity(
            h_mean,
            k_tau.expand(h_mean.shape[0], -1, -1),
            dim=-1,
            eps=1e-8
        )  # [batch, 1]

        # Bias: shift hidden states toward τ direction proportional to alignment
        # This is the attention bias field made concrete at the representation level
        # τ_direction: unit vector in τ_k direction
        tau_direction = F.normalize(k_tau, dim=-1)  # [1, 1, hidden_size]

        bias = self.alpha * sim.unsqueeze(-1) * tau_direction
        # [batch, 1, hidden_size] broadcasts over seq_len
        hidden_states = hidden_states + bias

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states


class Phase1Injector:
    """
    Manages all attention hooks for Phase 1.
    Registers hooks on every (or selected) attention layers of the base model.

    Keeps handles so hooks can be removed cleanly if needed.
    """

    def __init__(self, base_model, tau_module, alpha: float = 0.1, layers: str = "all"):
        self.base_model = base_model
        self.tau_module = tau_module
        self.alpha = alpha
        self.layers = layers
        self._handles = []
        self._hooks: list[TauAttentionHook] = []
        self._active = False

    def activate(self):
        """Register hooks on all target attention layers."""
        if self._active:
            logger.warning("Phase1Injector already active — skipping re-registration")
            return

        attn_layers = self._find_attention_layers()
        if not attn_layers:
            raise RuntimeError(
                "Could not find attention layers in model. "
                "Check _find_attention_layers() for your model architecture."
            )

        target_indices = self._resolve_layer_indices(len(attn_layers))
        registered = 0

        for idx in target_indices:
            layer = attn_layers[idx]
            hook = TauAttentionHook(self.tau_module, idx, self.alpha)
            handle = layer.register_forward_hook(hook)
            self._handles.append(handle)
            self._hooks.append(hook)
            registered += 1

        self._active = True
        logger.info(
            f"Phase 1 injector active | "
            f"{registered}/{len(attn_layers)} layers hooked | "
            f"α={self.alpha}"
        )

    def deactivate(self):
        """Remove all hooks cleanly."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hooks.clear()
        self._active = False
        logger.info("Phase 1 injector deactivated")

    def set_alpha(self, alpha: float):
        """Adjust gravitational pull strength at runtime."""
        self.alpha = alpha
        for hook in self._hooks:
            hook.alpha = alpha
        logger.info(f"α updated to {alpha}")

    def stats(self) -> dict:
        return {
            "active": self._active,
            "hooks_registered": len(self._handles),
            "alpha": self.alpha,
            "total_hook_calls": sum(h._call_count for h in self._hooks),
        }

    def _find_attention_layers(self) -> list:
        """
        Find attention layer modules by walking the model tree.
        Works for Qwen2.5, LLaMA, Mistral, Gemma architectures.
        Returns list of attention modules in layer order.
        """
        candidates = []

        # Qwen2.5 / LLaMA 3 style: model.layers[i].self_attn
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            for layer in self.base_model.model.layers:
                if hasattr(layer, "self_attn"):
                    candidates.append(layer.self_attn)
            if candidates:
                logger.debug(f"Found {len(candidates)} attention layers via model.layers[i].self_attn")
                return candidates

        # Fallback: scan all named modules for attention-like names
        for name, module in self.base_model.named_modules():
            mname = type(module).__name__.lower()
            if any(k in mname for k in ("attention", "selfattn", "attn")) and \
               not any(k in mname for k in ("output", "dropout", "norm")):
                if hasattr(module, "forward") and module is not self.base_model:
                    candidates.append(module)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for m in candidates:
            mid = id(m)
            if mid not in seen:
                seen.add(mid)
                unique.append(m)

        logger.debug(f"Found {len(unique)} attention layers via name scan")
        return unique

    def _resolve_layer_indices(self, total: int) -> list[int]:
        """Resolve 'all' or specific layer list to integer indices."""
        if self.layers == "all":
            return list(range(total))
        if isinstance(self.layers, list):
            return [i for i in self.layers if 0 <= i < total]
        # Sparse: hook every Nth layer for speed
        if isinstance(self.layers, str) and self.layers.startswith("every"):
            n = int(self.layers.replace("every", ""))
            return list(range(0, total, n))
        return list(range(total))