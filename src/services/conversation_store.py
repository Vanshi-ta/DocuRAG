

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from config import BASE_DIR

DEFAULT_CONVERSATIONS_DIR = BASE_DIR / "data" / "conversations"
_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_TITLE_MAX_CHARS = 60
DEFAULT_TITLE = "New conversation"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StoredSource:
    source_filename: str
    page_number: int
    similarity_score: float
    chunk_id: str
    chunk_text: str


@dataclass
class StoredMessage:
    id: str
    role: str                        
    content: str
    created_at: str
    sources: list[StoredSource] | None = None
    used_llm: bool | None = None
    is_error: bool = False


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[StoredMessage] = field(default_factory=list)


@dataclass
class ConversationSummary:
    id: str
    title: str
    updated_at: str
    message_count: int


def sources_from_chunks(chunks: Iterable[Any]) -> list[StoredSource]:
    
    return [
        StoredSource(
            source_filename=c.source_filename,
            page_number=c.page_number,
            similarity_score=float(c.similarity_score),
            chunk_id=c.chunk_id,
            chunk_text=c.chunk_text,
        )
        for c in chunks
    ]


class ConversationStore:
    def __init__(self, directory: Path = DEFAULT_CONVERSATIONS_DIR):
        self.directory = Path(directory)

                                                                             
    def _path(self, conversation_id: str) -> Path:
        if not _ID_PATTERN.fullmatch(conversation_id or ""):
            raise ValueError(f"Invalid conversation id: {conversation_id!r}")
        return self.directory / f"{conversation_id}.json"

    def _write(self, conv: Conversation) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(conv.id)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(conv), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)                                                    

    @staticmethod
    def _from_dict(raw: dict[str, Any]) -> Conversation:
        messages = []
        for m in raw.get("messages", []):
            sources = m.get("sources")
            messages.append(
                StoredMessage(
                    id=m["id"],
                    role=m["role"],
                    content=m["content"],
                    created_at=m["created_at"],
                    sources=[StoredSource(**s) for s in sources] if sources is not None else None,
                    used_llm=m.get("used_llm"),
                    is_error=m.get("is_error", False),
                )
            )
        return Conversation(
            id=raw["id"],
            title=raw["title"],
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
            messages=messages,
        )

                                                                               
    def create(self, title: str | None = None) -> Conversation:
        now = _now()
        conv = Conversation(id=uuid.uuid4().hex, title=title or DEFAULT_TITLE, created_at=now, updated_at=now)
        self._write(conv)
        return conv

    def get(self, conversation_id: str) -> Conversation:
        path = self._path(conversation_id)
        if not path.exists():
            raise FileNotFoundError(f"No conversation {conversation_id}")
        with open(path, "r", encoding="utf-8") as f:
            return self._from_dict(json.load(f))

    def list(self) -> list[ConversationSummary]:
        
        if not self.directory.exists():
            return []
        summaries = []
        for path in self.directory.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                summaries.append(
                    ConversationSummary(
                        id=raw["id"],
                        title=raw["title"],
                        updated_at=raw["updated_at"],
                        message_count=len(raw.get("messages", [])),
                    )
                )
            except (OSError, ValueError, KeyError):
                continue
        return sorted(summaries, key=lambda s: s.updated_at, reverse=True)

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Iterable[Any] | None = None,
        used_llm: bool | None = None,
        is_error: bool = False,
    ) -> StoredMessage:
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        conv = self.get(conversation_id)
        message = StoredMessage(
            id=uuid.uuid4().hex,
            role=role,
            content=content,
            created_at=_now(),
            sources=sources_from_chunks(sources) if sources is not None else None,
            used_llm=used_llm,
            is_error=is_error,
        )
        conv.messages.append(message)
        conv.updated_at = message.created_at
                                                  
        if conv.title == DEFAULT_TITLE and role == "user":
            first_line = content.strip().splitlines()[0] if content.strip() else ""
            if first_line:
                conv.title = first_line[:_TITLE_MAX_CHARS] + ("…" if len(first_line) > _TITLE_MAX_CHARS else "")
        self._write(conv)
        return message

    def rename(self, conversation_id: str, title: str) -> Conversation:
        title = title.strip()
        if not title:
            raise ValueError("title must be non-empty")
        conv = self.get(conversation_id)
        conv.title = title
        conv.updated_at = _now()
        self._write(conv)
        return conv

    def delete(self, conversation_id: str) -> bool:
        path = self._path(conversation_id)
        if path.exists():
            path.unlink()
            return True
        return False
