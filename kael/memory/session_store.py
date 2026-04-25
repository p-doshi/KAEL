"""
KAEL Session Store
SQLite-backed session logging. Every input/output gets saved here.
This is the raw material M reads from during consolidation.
"""

import sqlite3
import json
import time
import uuid
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict

from config import cfg


@dataclass
class Session:
    """A single conversation session."""
    session_id: str
    timestamp: float
    user_input: str
    model_output: str
    tau_snapshot: Optional[list]        # τ vector at start of session (serialized)
    gate_value: Optional[float]         # g_t computed after this session
    importance_score: Optional[float]   # Assigned by importance scorer
    novelty_score: Optional[float]      # How novel was this session's content
    domain: Optional[str]               # Inferred domain (science/math/philosophy/etc)
    human_feedback: Optional[str]       # "integrate" / "reject" / "modify" / None
    metadata: Optional[dict]            # Catch-all for extra fields
    session_embedding: Optional[list]   # Mean-pool hidden state (for FAISS)

    @classmethod
    def new(cls, user_input: str, model_output: str) -> "Session":
        return cls(
            session_id=str(uuid.uuid4()),
            timestamp=time.time(),
            user_input=user_input,
            model_output=model_output,
            tau_snapshot=None,
            gate_value=None,
            importance_score=None,
            novelty_score=None,
            domain=None,
            human_feedback=None,
            metadata=None,
            session_embedding=None,
        )


class SessionStore:
    """
    SQLite session store.
    Thread-safe for single-user use. For multi-user, add connection pooling.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id      TEXT PRIMARY KEY,
        timestamp       REAL NOT NULL,
        user_input      TEXT NOT NULL,
        model_output    TEXT NOT NULL,
        tau_snapshot    TEXT,           -- JSON serialized list[float]
        gate_value      REAL,
        importance_score REAL,
        novelty_score   REAL,
        domain          TEXT,
        human_feedback  TEXT,
        metadata        TEXT,           -- JSON
        session_embedding TEXT          -- JSON serialized list[float]
    );

    CREATE INDEX IF NOT EXISTS idx_timestamp ON sessions(timestamp);
    CREATE INDEX IF NOT EXISTS idx_importance ON sessions(importance_score);
    CREATE INDEX IF NOT EXISTS idx_gate ON sessions(gate_value);
    CREATE INDEX IF NOT EXISTS idx_domain ON sessions(domain);

    CREATE TABLE IF NOT EXISTS tau_snapshots (
        snapshot_id     TEXT PRIMARY KEY,
        timestamp       REAL NOT NULL,
        session_count   INTEGER NOT NULL,
        tau_vector      TEXT NOT NULL,   -- JSON list[float]
        notes           TEXT
    );

    CREATE TABLE IF NOT EXISTS human_feedback (
        feedback_id     TEXT PRIMARY KEY,
        session_id      TEXT NOT NULL,
        timestamp       REAL NOT NULL,
        decision        TEXT NOT NULL,   -- integrate / reject / modify
        proposed_delta  TEXT,            -- JSON list[float] — the Δτ being reviewed
        modified_delta  TEXT,            -- JSON list[float] — if modified by human
        notes           TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or cfg.memory.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")    # Better concurrent read performance
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)

    # ── Session CRUD ──────────────────────────────────────────────────────────

    def save_session(self, session: Session) -> str:
        """Save a session. Returns session_id."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions VALUES (
                    :session_id, :timestamp, :user_input, :model_output,
                    :tau_snapshot, :gate_value, :importance_score, :novelty_score,
                    :domain, :human_feedback, :metadata, :session_embedding
                )
            """, {
                **asdict(session),
                "tau_snapshot": json.dumps(session.tau_snapshot) if session.tau_snapshot else None,
                "metadata": json.dumps(session.metadata) if session.metadata else None,
                "session_embedding": json.dumps(session.session_embedding) if session.session_embedding else None,
            })
        return session.session_id

    def update_session(self, session_id: str, **kwargs):
        """Update specific fields of an existing session."""
        # Serialize any list fields
        for key in ("tau_snapshot", "session_embedding"):
            if key in kwargs and isinstance(kwargs[key], list):
                kwargs[key] = json.dumps(kwargs[key])
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = json.dumps(kwargs["metadata"])

        set_clause = ", ".join(f"{k} = :{k}" for k in kwargs)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE sessions SET {set_clause} WHERE session_id = :session_id",
                {**kwargs, "session_id": session_id}
            )

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._row_to_session(row) if row else None

    def get_recent_sessions(self, n: int = 20) -> list[Session]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def get_high_importance_sessions(self, threshold: float = 0.6, limit: int = 100) -> list[Session]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM sessions
                WHERE importance_score >= ?
                ORDER BY importance_score DESC
                LIMIT ?
            """, (threshold, limit)).fetchall()
        return [self._row_to_session(r) for r in rows]

    def get_contradiction_sessions(self) -> list[Session]:
        """Sessions where gate fired the human feedback loop."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM sessions
                WHERE gate_value <= ?
                ORDER BY timestamp DESC
            """, (cfg.gate.contradiction_threshold,)).fetchall()
        return [self._row_to_session(r) for r in rows]

    def get_unfeedback_sessions(self) -> list[Session]:
        """Contradiction sessions awaiting human review."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM sessions
                WHERE gate_value <= ? AND human_feedback IS NULL
                ORDER BY timestamp ASC
            """, (cfg.gate.contradiction_threshold,)).fetchall()
        return [self._row_to_session(r) for r in rows]

    def count_sessions(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    # ── τ snapshots ───────────────────────────────────────────────────────────

    def save_tau_snapshot(self, tau_vector: list[float], session_count: int, notes: str = ""):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO tau_snapshots VALUES (?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), time.time(), session_count, json.dumps(tau_vector), notes))

    def get_tau_snapshots(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tau_snapshots ORDER BY session_count ASC"
            ).fetchall()
        return [
            {
                "snapshot_id": r["snapshot_id"],
                "timestamp": r["timestamp"],
                "session_count": r["session_count"],
                "tau_vector": json.loads(r["tau_vector"]),
                "notes": r["notes"],
            }
            for r in rows
        ]

    # ── Human feedback ────────────────────────────────────────────────────────

    def save_human_feedback(
        self,
        session_id: str,
        decision: str,
        proposed_delta: Optional[list[float]] = None,
        modified_delta: Optional[list[float]] = None,
        notes: str = "",
    ):
        assert decision in ("integrate", "reject", "modify"), \
            f"decision must be integrate/reject/modify, got {decision}"

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO human_feedback VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                session_id,
                time.time(),
                decision,
                json.dumps(proposed_delta) if proposed_delta else None,
                json.dumps(modified_delta) if modified_delta else None,
                notes,
            ))
            # Update the session record
            conn.execute(
                "UPDATE sessions SET human_feedback = ? WHERE session_id = ?",
                (decision, session_id)
            )

    def get_human_feedback(self, session_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM human_feedback WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1",
                (session_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "decision": row["decision"],
            "proposed_delta": json.loads(row["proposed_delta"]) if row["proposed_delta"] else None,
            "modified_delta": json.loads(row["modified_delta"]) if row["modified_delta"] else None,
            "notes": row["notes"],
        }

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            flagged = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE gate_value <= ?",
                (cfg.gate.contradiction_threshold,)
            ).fetchone()[0]
            reviewed = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE human_feedback IS NOT NULL"
            ).fetchone()[0]
            avg_gate = conn.execute(
                "SELECT AVG(gate_value) FROM sessions WHERE gate_value IS NOT NULL"
            ).fetchone()[0]
            avg_importance = conn.execute(
                "SELECT AVG(importance_score) FROM sessions WHERE importance_score IS NOT NULL"
            ).fetchone()[0]
            domains = conn.execute(
                "SELECT domain, COUNT(*) as cnt FROM sessions WHERE domain IS NOT NULL GROUP BY domain ORDER BY cnt DESC"
            ).fetchall()

        return {
            "total_sessions": total,
            "flagged_for_review": flagged,
            "reviewed": reviewed,
            "pending_review": flagged - reviewed,
            "avg_gate_value": round(avg_gate, 4) if avg_gate else None,
            "avg_importance": round(avg_importance, 4) if avg_importance else None,
            "domain_distribution": {r["domain"]: r["cnt"] for r in domains},
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        d = dict(row)
        d["tau_snapshot"] = json.loads(d["tau_snapshot"]) if d["tau_snapshot"] else None
        d["session_embedding"] = json.loads(d["session_embedding"]) if d["session_embedding"] else None
        d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else None
        return Session(**d)
