"""
KAEL Phase 0 Tests
Run these before anything else to verify the infrastructure is solid.
No model loading required for most tests — they test the plumbing.

Usage:
    python -m pytest tests/test_phase0.py -v
    python tests/test_phase0.py   # without pytest
"""

import json
import time
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import KAELConfig, ModelConfig, TauConfig, GateConfig, ConsolidationConfig
from memory.session_store import Session, SessionStore


class TestConfig(unittest.TestCase):
    """Config validation."""

    def test_default_config_validates(self):
        cfg = KAELConfig()
        cfg.validate()

    def test_tau_dims_sum(self):
        cfg = KAELConfig()
        total = cfg.tau.epistemic_dim + cfg.tau.dispositional_dim + cfg.tau.relational_dim
        self.assertEqual(total, cfg.tau.dim, f"τ sub-dims {total} != tau.dim {cfg.tau.dim}")

    def test_gate_threshold_ordering(self):
        cfg = KAELConfig()
        self.assertLess(cfg.gate.contradiction_threshold, cfg.gate.integration_threshold)

    def test_consolidation_weights_sum(self):
        cfg = KAELConfig()
        total = cfg.consolidation.lambda_consolidation + cfg.consolidation.lambda_coherence
        self.assertAlmostEqual(total, 1.0)


class TestSessionStore(unittest.TestCase):
    """Session store CRUD and queries."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        cfg = KAELConfig()
        cfg.memory.db_path = Path(self.tmpdir) / "test.db"
        self.store = SessionStore(cfg.memory.db_path)

    def test_save_and_retrieve(self):
        session = Session.new("What is gravity?", "Gravity is a fundamental force...")
        sid = self.store.save_session(session)
        retrieved = self.store.get_session(sid)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.user_input, "What is gravity?")
        self.assertEqual(retrieved.model_output, "Gravity is a fundamental force...")

    def test_tau_snapshot_serialization(self):
        session = Session.new("test", "response")
        session.tau_snapshot = [0.1, 0.2, 0.3, -0.1, 0.05]
        self.store.save_session(session)
        retrieved = self.store.get_session(session.session_id)
        self.assertEqual(len(retrieved.tau_snapshot), 5)
        self.assertAlmostEqual(retrieved.tau_snapshot[0], 0.1)

    def test_session_embedding_serialization(self):
        session = Session.new("test", "response")
        session.session_embedding = [float(i) * 0.01 for i in range(10)]
        self.store.save_session(session)
        retrieved = self.store.get_session(session.session_id)
        self.assertEqual(len(retrieved.session_embedding), 10)

    def test_update_gate_value(self):
        session = Session.new("test", "response")
        self.store.save_session(session)
        self.store.update_session(session.session_id, gate_value=0.15)
        retrieved = self.store.get_session(session.session_id)
        self.assertAlmostEqual(retrieved.gate_value, 0.15)

    def test_get_recent_sessions(self):
        for i in range(5):
            time.sleep(0.01)  # Ensure distinct timestamps
            self.store.save_session(Session.new(f"question {i}", f"answer {i}"))
        recent = self.store.get_recent_sessions(3)
        self.assertEqual(len(recent), 3)
        # Should be newest first
        self.assertIn("question 4", recent[0].user_input)

    def test_count_sessions(self):
        self.assertEqual(self.store.count_sessions(), 0)
        self.store.save_session(Session.new("q", "a"))
        self.assertEqual(self.store.count_sessions(), 1)

    def test_stats(self):
        session = Session.new("test", "response")
        session.gate_value = 0.8
        session.importance_score = 0.6
        session.domain = "physics"
        self.store.save_session(session)
        stats = self.store.get_stats()
        self.assertEqual(stats["total_sessions"], 1)
        self.assertIn("domain_distribution", stats)

    def test_contradiction_sessions(self):
        # Low gate → contradiction
        s1 = Session.new("contradiction", "response")
        s1.gate_value = 0.1
        self.store.save_session(s1)
        # High gate → not contradiction
        s2 = Session.new("normal", "response")
        s2.gate_value = 0.9
        self.store.save_session(s2)

        contradictions = self.store.get_contradiction_sessions()
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0].session_id, s1.session_id)

    def test_human_feedback(self):
        session = Session.new("test", "response")
        self.store.save_session(session)

        self.store.save_human_feedback(
            session_id=session.session_id,
            decision="integrate",
            notes="This is consistent with prior understanding",
        )
        feedback = self.store.get_human_feedback(session.session_id)
        self.assertIsNotNone(feedback)
        self.assertEqual(feedback["decision"], "integrate")

        # Verify session record updated
        retrieved = self.store.get_session(session.session_id)
        self.assertEqual(retrieved.human_feedback, "integrate")

    def test_tau_snapshot_storage(self):
        tau = [float(i) * 0.001 for i in range(100)]
        self.store.save_tau_snapshot(tau, session_count=10, notes="test snapshot")
        snapshots = self.store.get_tau_snapshots()
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["session_count"], 10)
        self.assertEqual(len(snapshots[0]["tau_vector"]), 100)

    def test_invalid_feedback_decision(self):
        session = Session.new("test", "response")
        self.store.save_session(session)
        with self.assertRaises(AssertionError):
            self.store.save_human_feedback(session.session_id, decision="wrong_value")

    def test_metadata_json(self):
        session = Session.new("test", "response")
        session.metadata = {"tokens": 42, "elapsed": 1.5, "nested": {"key": "val"}}
        self.store.save_session(session)
        retrieved = self.store.get_session(session.session_id)
        self.assertEqual(retrieved.metadata["tokens"], 42)
        self.assertEqual(retrieved.metadata["nested"]["key"], "val")


class TestTauEmbedding(unittest.TestCase):
    """TauEmbedding without GPU — CPU only."""

    def setUp(self):
        try:
            import torch
            self.torch = torch
        except ImportError:
            self.skipTest("PyTorch not installed")

    def test_tau_init(self):
        from core.model import TauEmbedding
        cfg = KAELConfig()
        cfg.model.device = "cpu"
        tau = TauEmbedding(cfg)
        self.assertEqual(tau.tau.shape[0], cfg.tau.dim)

    def test_tau_slices(self):
        from core.model import TauEmbedding
        cfg = KAELConfig()
        cfg.model.device = "cpu"
        tau = TauEmbedding(cfg)
        e = tau.tau_epistemic
        d = tau.tau_dispositional
        r = tau.tau_relational
        self.assertEqual(e.shape[0], cfg.tau.epistemic_dim)
        self.assertEqual(d.shape[0], cfg.tau.dispositional_dim)
        self.assertEqual(r.shape[0], cfg.tau.relational_dim)
        # Should cover full τ without overlap
        self.assertEqual(e.shape[0] + d.shape[0] + r.shape[0], cfg.tau.dim)

    def test_snapshot_roundtrip(self):
        from core.model import TauEmbedding
        cfg = KAELConfig()
        cfg.model.device = "cpu"
        tau = TauEmbedding(cfg)
        original = tau.snapshot()
        tau.load_snapshot(original)
        restored = tau.snapshot()
        self.assertEqual(len(original), len(restored))
        for a, b in zip(original, restored):
            self.assertAlmostEqual(a, b, places=5)

    def test_norm(self):
        from core.model import TauEmbedding
        cfg = KAELConfig()
        cfg.model.device = "cpu"
        tau = TauEmbedding(cfg)
        norm = tau.norm()
        self.assertGreater(norm, 0)
        self.assertIsInstance(norm, float)


def run_tests():
    """Run all tests with basic output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestConfig, TestSessionStore, TestTauEmbedding]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
