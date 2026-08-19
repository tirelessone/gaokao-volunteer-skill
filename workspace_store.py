"""Temporary JSON-file workspace store with Redis-like TTL semantics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable


class FileWorkspaceStore:
    """Persist temporary values in one text file until a Redis backend is available."""

    def __init__(
        self,
        path: str | Path,
        *,
        ttl_seconds: int = 30 * 60,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self._now = now
        self._lock = threading.RLock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            document = self._read_document()
            changed = self._remove_expired(document)
            record = document["entries"].get(key)
            if changed:
                self._write_document(document)
            if record is None:
                return None
            value = record.get("value")
            return dict(value) if isinstance(value, dict) else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            document = self._read_document()
            self._remove_expired(document)
            now = self._now()
            document["entries"][key] = {
                "updated_at": now,
                "expires_at": now + self.ttl_seconds,
                "value": value,
            }
            self._write_document(document)

    def delete(self, key: str) -> None:
        with self._lock:
            document = self._read_document()
            if document["entries"].pop(key, None) is not None:
                self._write_document(document)

    def metadata(self, key: str) -> dict[str, Any]:
        with self._lock:
            document = self._read_document()
            changed = self._remove_expired(document)
            record = document["entries"].get(key)
            if changed:
                self._write_document(document)
            if record is None:
                return {"available": False, "ttl_seconds": 0}
            return {
                "available": True,
                "ttl_seconds": max(0, int(record["expires_at"] - self._now())),
                "updated_at": record.get("updated_at"),
            }

    def _read_document(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "entries": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "entries": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
            return {"version": 1, "entries": {}}
        return payload

    def _write_document(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def _remove_expired(self, document: dict[str, Any]) -> bool:
        now = self._now()
        expired = [
            key for key, record in document["entries"].items()
            if not isinstance(record, dict) or float(record.get("expires_at", 0)) <= now
        ]
        for key in expired:
            document["entries"].pop(key, None)
        return bool(expired)
