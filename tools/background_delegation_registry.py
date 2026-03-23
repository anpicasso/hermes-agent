"""
Background Delegation Registry — Thread-safe registry for fire-and-forget subagent tasks.

Tracks background delegations spawned via delegate_task(background=True), providing:
  - Session tracking (running / finished)
  - Result storage (populated by the child agent thread when done)
  - Thread-safe drain by the gateway for result delivery

Usage:
    from tools.background_delegation_registry import bg_delegation_registry

    # Spawn: delegate_task(background=True) calls register()
    session = bg_delegation_registry.register(prompt, source, task_id, toolsets)

    # Drain: gateway calls drain_pending() after agent run finishes
    pending = bg_delegation_registry.drain_pending()

    # Watcher polls: session.done_event.wait() → session.result available
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Limits
FINISHED_TTL_SECONDS = 3600  # Keep finished delegations for 1 hour
MAX_DELEGATIONS = 128  # Max concurrent tracked delegations (LRU pruning)


@dataclass
class BackgroundDelegationSession:
    """A tracked background delegation with result storage."""

    id: str  # Unique session ID ("bg_delegate_xxxxxxxxxx")
    prompt: str  # The goal/context prompt
    source: Any  # SessionSource — where to deliver the result
    task_id: str = ""  # Task identifier
    toolsets: List[str] = field(
        default_factory=list
    )  # Enabled toolsets for the child agent
    started_at: float = 0.0  # time.time() of spawn
    finished_at: float = 0.0  # time.time() of completion (0 if still running)
    done_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[Dict[str, Any]] = None  # Populated when child finishes
    error: Optional[str] = None  # Error message if child raised
    _lock: threading.Lock = field(default_factory=threading.Lock)


class BackgroundDelegationRegistry:
    """
    Thread-safe registry of running and finished background delegations.

    Accessed from:
      - Executor threads (delegate_tool background spawn)
      - Gateway asyncio loop (drain_pending after agent run)
      - Watcher tasks (poll for done, send result)
    """

    def __init__(self):
        self._running: Dict[str, BackgroundDelegationSession] = {}
        self._finished: Dict[str, BackgroundDelegationSession] = {}
        self._lock = threading.Lock()
        # Side-channel for gateway pickup (populated after child spawns)
        self.pending: List[BackgroundDelegationSession] = []

    # ----- Spawn -----

    def register(
        self,
        prompt: str,
        source: Any,
        task_id: str,
        toolsets: List[str],
    ) -> BackgroundDelegationSession:
        """
        Register a new background delegation session.

        Called from delegate_tool's background thread before it starts.
        The session is added to _running immediately; when the thread
        is about to exit it calls mark_done(result).

        Returns the session.
        """
        session = BackgroundDelegationSession(
            id=f"bg_delegate_{uuid.uuid4().hex[:10]}",
            prompt=prompt,
            source=source,
            task_id=task_id,
            toolsets=toolsets,
            started_at=time.time(),
        )

        with self._lock:
            self._prune_if_needed()
            self._running[session.id] = session

        with self._lock:
            self.pending.append(session)

        return session

    def mark_done(
        self,
        session_id: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """
        Mark a delegation as finished and store its result.

        Called from the child agent thread when it completes.
        """
        session = None
        with self._lock:
            session = self._running.pop(session_id, None)

        if session is None:
            logger.warning("mark_done called on unknown session %s", session_id)
            return

        session.finished_at = time.time()
        session.result = result
        session.error = error
        session.done_event.set()

        with self._lock:
            self._finished[session.id] = session

        logger.debug(
            "Background delegation %s finished (error=%s)",
            session_id,
            error is not None,
        )

    # ----- Gateway Drain -----

    def drain_pending(self) -> List[BackgroundDelegationSession]:
        """
        Atomically drain and return all pending delegation sessions.

        Called from the gateway's async loop after agent.run_conversation()
        returns.  The sessions are removed from `pending` so they are not
        returned again on the next drain call.
        """
        with self._lock:
            sessions = list(self.pending)
            self.pending.clear()
        return sessions

    # ----- Query -----

    def get(self, session_id: str) -> Optional[BackgroundDelegationSession]:
        """Get a session by ID (running or finished)."""
        with self._lock:
            return self._running.get(session_id) or self._finished.get(session_id)

    def is_running(self, session_id: str) -> bool:
        """Return True if the session is still running."""
        with self._lock:
            return session_id in self._running

    def wait(self, session_id: str, timeout: float = None) -> Dict[str, Any]:
        """
        Block until a delegation finishes (or timeout).

        Returns {"status": "finished", "result": ...} or
                {"status": "timeout", "session_id": ...}.
        """
        session = self.get(session_id)
        if session is None:
            return {
                "status": "not_found",
                "error": f"No delegation with ID {session_id}",
            }

        finished = session.done_event.wait(timeout=timeout if timeout else 3600)
        if not finished:
            return {"status": "timeout", "session_id": session_id}

        return {
            "status": "finished",
            "result": session.result,
            "error": session.error,
            "duration_seconds": round(session.finished_at - session.started_at, 2),
        }

    def list_running(self) -> List[Dict[str, Any]]:
        """List all currently-running background delegations."""
        with self._lock:
            sessions = list(self._running.values())

        return [
            {
                "id": s.id,
                "task_id": s.task_id,
                "prompt_preview": s.prompt[:80],
                "uptime_seconds": int(time.time() - s.started_at),
                "started_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(s.started_at)
                ),
            }
            for s in sessions
        ]

    # ----- Cleanup / Pruning -----

    def _prune_if_needed(self):
        """Remove oldest finished delegations if over MAX_DELEGATIONS."""
        if len(self._finished) < MAX_DELEGATIONS:
            return

        now = time.time()
        expired = [
            sid
            for sid, s in self._finished.items()
            if (now - s.started_at) > FINISHED_TTL_SECONDS
        ]
        for sid in expired:
            del self._finished[sid]

        # If still over limit, remove oldest finished
        total = len(self._running) + len(self._finished)
        if total >= MAX_DELEGATIONS and self._finished:
            oldest_id = min(
                self._finished, key=lambda sid: self._finished[sid].started_at
            )
            del self._finished[oldest_id]


# Module-level singleton
bg_delegation_registry = BackgroundDelegationRegistry()
