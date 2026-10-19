"""SQLite state for the local, single-user Agent service."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
              id TEXT PRIMARY KEY, conversation_id TEXT, status TEXT NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              result TEXT NOT NULL DEFAULT '', elapsed_ms REAL);
            CREATE TABLE IF NOT EXISTS events (
              seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
              created_at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS events_task ON events(task_id, seq);
            CREATE TABLE IF NOT EXISTS conversations (
              id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, conversation_id):
        task_id = uuid4().hex
        with self.connect() as db:
            db.execute(
                "INSERT INTO tasks(id,conversation_id,status,created_at,updated_at) VALUES(?,?,?,?,?)",
                (task_id, conversation_id, "queued", now(), now()),
            )
        return task_id

    def update(self, task_id, status, result="", elapsed_ms=None):
        if status not in {
            "queued",
            "running",
            "completed",
            "failed",
            "cancelled",
            "interrupted",
        }:
            raise ValueError("Invalid task status")
        with self.connect() as db:
            db.execute(
                "UPDATE tasks SET status=?,result=?,elapsed_ms=?,updated_at=? WHERE id=?",
                (status, result[:100000], elapsed_ms, now(), task_id),
            )

    def event(self, task_id, kind, payload):
        with self.connect() as db:
            db.execute(
                "INSERT INTO events(task_id,created_at,kind,payload) VALUES(?,?,?,?)",
                (task_id, now(), kind, json.dumps(payload, ensure_ascii=False)),
            )

    def get(self, task_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["events"] = [
                dict(e, payload=json.loads(e["payload"]))
                for e in db.execute(
                    "SELECT * FROM events WHERE task_id=? ORDER BY seq", (task_id,)
                )
            ]
            return result

    def list(self, limit=20):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            ]

    def recover(self):
        with self.connect() as db:
            db.execute(
                "UPDATE tasks SET status='interrupted',updated_at=? WHERE status IN ('queued','running')",
                (now(),),
            )

    def save_conversation(self, conversation):
        # Reasoning is transient; persist only user-visible messages and artifacts.
        payload = dict(
            conversation,
            messages=[
                {k: v for k, v in m.items() if k != "reasoning_content"}
                for m in conversation["messages"]
            ],
        )
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO conversations VALUES(?,?)",
                (conversation["id"], json.dumps(payload, ensure_ascii=False)),
            )

    def conversations(self):
        with self.connect() as db:
            return {
                r["id"]: json.loads(r["payload"])
                for r in db.execute("SELECT * FROM conversations")
            }
