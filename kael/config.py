"""
KAEL Configuration
Kinetic Adaptive Epistemic Lattice

All hyperparameters and paths live here.
Change these before running — nothing is hardcoded elsewhere.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent


@dataclass
class ModelConfig:
    # Base model — frozen at inference, never modified
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # For development on smaller hardware, swap to:
    # "Qwen/Qwen2.5-7B-Instruct"       (~14GB VRAM)
    # "Qwen/Qwen2.5-3B-Instruct"        (~6GB VRAM, fast iteration)
    # "mistralai/Mistral-7B-Instruct-v0.3"

    dtype: str = "bfloat16"           # bfloat16 preferred, float16 if needed
    device: str = "cuda"
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    load_in_4bit: bool = True        # Set True if VRAM tight
    load_in_8bit: bool = False

    # Context window — Qwen2.5 supports 128k, we use 8k for speed
    max_context_length: int = 8192


@dataclass
class TauConfig:
    # Trajectory embedding τ ∈ ℝ^d
    # d matches model hidden size for clean projection
    dim: int = 2048                    # Match Qwen2.5-14B hidden dim (5120 for 14B, 4096 for 7B)
    # NOTE: Qwen2.5-14B hidden_size = 5120. Qwen2.5-7B = 3584. Qwen2.5-3B = 2048
    # Set this to match your chosen model's hidden_size

    init_scale: float = 0.01           # Small random init, not zeros (zeros = no signal)
    attention_alpha: float = 0.1       # Gravitational pull strength — start small, tune up
    prefix_layers: str = "all"         # "all" or list of layer indices e.g. [0,8,16,24]

    # Sub-embedding dimensions (must sum to dim)
    epistemic_dim: int = 683          # Knowledge frontier shape
    dispositional_dim: int = 682      # Reasoning signature
    relational_dim: int = 683         # Collaborative formation history
    # 683 + 682 + 683 = 2048


@dataclass
class GateConfig:
    # Gate g_t ∈ [0,1] — stability-plasticity controller
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1

    # Thresholds
    contradiction_threshold: float = 0.2    # g_t below this → trigger human loop
    integration_threshold: float = 0.8      # g_t above this → integrate fully
    # Between 0.2 and 0.8 → partial integration (novelty expansion zone)

    # Training
    lr: float = 3e-4
    batch_size: int = 64
    epochs: int = 20


@dataclass
class ConsolidationConfig:
    # Consolidation network f: (τ_t, s_t, θ) → Δτ
    # Small transformer, not MLP — handles sequential session structure
    num_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 512
    dropout: float = 0.1

    # Loss weights — anneal based on novelty during training
    lambda_consolidation: float = 0.7
    lambda_coherence: float = 0.3

    # Directed growth
    alpha_coherence: float = 0.5       # Pull toward internal consistency
    beta_novelty: float = 0.5          # Pull toward frontier

    lr: float = 1e-4
    batch_size: int = 32
    warmup_steps: int = 500


@dataclass
class MemoryConfig:
    # External memory store M
    db_path: Path = ROOT / "memory" / "sessions.db"
    faiss_path: Path = ROOT / "memory" / "session_index.faiss"

    # FAISS config
    embedding_dim: int = 2048           # Mean-pool of last hidden state
    index_type: str = "IVFFlat"         # IVFFlat for recall, Flat for exact (small scale)
    nlist: int = 100                    # Number of Voronoi cells (IVFFlat)
    nprobe: int = 10                    # Cells to search at query time

    # Memory management
    max_sessions: int = 1000
    importance_prune_threshold: float = 0.2
    top_k_retrieval: int = 5            # Sessions to retrieve during consolidation

    # Importance scoring thresholds
    high_importance_gate_threshold: float = 0.3   # g_t below this → high importance
    high_importance_novelty_threshold: float = 0.7 # novelty above this → high importance


@dataclass
class LoggingConfig:
    log_dir: Path = ROOT / "logs"
    checkpoint_dir: Path = ROOT / "checkpoints"

    # Weights & Biases — set to None to disable
    wandb_project: Optional[str] = "kael"
    wandb_entity: Optional[str] = None  # Your wandb username

    # How often to save τ snapshots (for trajectory visualization)
    tau_snapshot_every_n_sessions: int = 5

    # Console logging level
    log_level: str = "INFO"


@dataclass
class EvalConfig:
    # Benchmarks to run at baseline and after each phase
    run_gsm8k: bool = True
    run_mmlu: bool = True
    run_frontier_qa: bool = True        # Our custom paper Q&A set
    run_consistency: bool = True        # Cross-framing reasoning consistency

    gsm8k_sample_size: int = 100        # Full = 1319, 100 for speed
    mmlu_subjects: list = field(default_factory=lambda: [
        "abstract_algebra",
        "college_physics",
        "machine_learning",
        "philosophy",
        "astronomy",
    ])
    mmlu_sample_per_subject: int = 50


@dataclass
class KAELConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tau: TauConfig = field(default_factory=TauConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Global seed
    seed: int = 42

    def validate(self):
        """Catch common config errors before they cause cryptic crashes."""
        assert self.tau.epistemic_dim + self.tau.dispositional_dim + self.tau.relational_dim == self.tau.dim, \
            f"τ sub-dims must sum to {self.tau.dim}, got {self.tau.epistemic_dim + self.tau.dispositional_dim + self.tau.relational_dim}"

        assert 0 < self.gate.contradiction_threshold < self.gate.integration_threshold < 1, \
            "Gate thresholds must satisfy: 0 < contradiction < integration < 1"

        assert self.consolidation.lambda_consolidation + self.consolidation.lambda_coherence == 1.0, \
            "Consolidation loss weights must sum to 1.0"

        # Create dirs if missing
        self.logging.log_dir.mkdir(parents=True, exist_ok=True)
        self.logging.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.memory.db_path.parent.mkdir(parents=True, exist_ok=True)

        return self


# Default config — import this everywhere
cfg = KAELConfig().validate()
