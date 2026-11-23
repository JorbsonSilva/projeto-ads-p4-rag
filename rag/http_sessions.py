from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from time import monotonic
from uuid import UUID, uuid4

from rag.session import Session, Turn


class ApiError(RuntimeError):
    def __init__(self, status, code, message, *, retryable=False):
        super().__init__(message)
        self.status, self.code, self.retryable = status, code, retryable


@dataclass(frozen=True)
class Limits:
    idle_seconds: int = 900
    sessions: int = 100
    turns: int = 20
    messages: int = 128
    concurrent: int = 4

    def __post_init__(self):
        if min(self.idle_seconds, self.sessions, self.turns, self.messages, self.concurrent) < 1:
            raise ValueError("Os limites de sessão devem ser positivos.")


@dataclass
class Conversation:
    owner: str
    session: Session
    last_activity: float
    busy: bool = False
    replies: dict[UUID, tuple[str, Turn | None]] = field(default_factory=dict)


class Conversations:
    def __init__(self, limits, *, clock=monotonic):
        self.limits = limits
        self.clock = clock
        self._items = {}
        self._lock = Lock()
        self._active = 0

    def _prune(self, now):
        expired = [
            key for key, item in self._items.items()
            if not item.busy and now - item.last_activity >= self.limits.idle_seconds
        ]
        for key in expired:
            del self._items[key]

    def prune(self):
        with self._lock:
            self._prune(self.clock())

    def create(self, owner, factory):
        with self._lock:
            now = self.clock()
            self._prune(now)
            if len(self._items) >= self.limits.sessions:
                raise ApiError(429, "SESSION_LIMIT", "Limite de conversas atingido.", retryable=True)
            key = uuid4()
            self._items[key] = Conversation(owner, factory(), now)
            return key, datetime.now(timezone.utc) + timedelta(seconds=self.limits.idle_seconds)

    def _owned(self, key, owner):
        self._prune(self.clock())
        item = self._items.get(key)
        if item is None or item.owner != owner:
            raise ApiError(404, "CONVERSATION_NOT_FOUND", "Conversa não encontrada.")
        return item

    def close(self, key, owner):
        with self._lock:
            item = self._owned(key, owner)
            if item.busy:
                raise ApiError(409, "TURN_IN_PROGRESS", "A conversa está processando um turno.", retryable=True)
            del self._items[key]

    def ask(self, key, owner, message_id, question):
        with self._lock:
            item = self._owned(key, owner)
            if message_id in item.replies:
                previous, result = item.replies[message_id]
                if question != previous:
                    raise ApiError(409, "MESSAGE_CONFLICT", "O ID de mensagem já foi usado com outra pergunta.")
                if result is not None:
                    if not item.busy:
                        item.last_activity = self.clock()
                    return result
            if item.busy:
                raise ApiError(409, "TURN_IN_PROGRESS", "A conversa está processando um turno.", retryable=True)
            if message_id not in item.replies and len(item.replies) >= self.limits.messages:
                raise ApiError(429, "MESSAGE_LIMIT", "Limite de mensagens atingido. Abra outra conversa.")
            if self._active >= self.limits.concurrent:
                raise ApiError(429, "CONCURRENCY_LIMIT", "Limite de chamadas simultâneas atingido.", retryable=True)
            item.busy = True
            item.replies[message_id] = (question, None)
            self._active += 1
        try:
            result = item.session.ask_result(question)
            with self._lock:
                item.replies[message_id] = (question, result)
            return result
        finally:
            with self._lock:
                item.last_activity = self.clock()
                item.busy = False
                self._active -= 1
