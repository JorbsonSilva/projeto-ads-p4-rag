import asyncio
import os
from contextlib import ExitStack, asynccontextmanager, suppress
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

import jwt
from chromadb.errors import ChromaError
from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, StringConstraints

from rag.http_auth import InternalAuth
from rag.http_modes import MODES, Mode, load_runtime
from rag.http_sessions import ApiError, Conversations, Limits
from rag.semantic_embeddings import EmbeddingUnavailable
from rag.session import ProviderError

PREFIX = "/internal/v1"


class CreateConversation(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messageId: UUID
    question: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=1000)]


class ConversationResponse(BaseModel):
    conversationId: UUID
    mode: Mode
    expiresAt: datetime


class EvidenceResponse(BaseModel):
    id: str
    corpus: str
    corpusVersion: int


class AnswerResponse(BaseModel):
    conversationId: UUID
    messageId: UUID
    mode: Mode
    outcome: Literal["answered", "no_evidence"]
    answer: str
    evidence: list[EvidenceResponse]


def _limits_from_env():
    defaults = Limits()
    return Limits(
        idle_seconds=int(os.environ.get("RAG_HTTP_IDLE_SECONDS", defaults.idle_seconds)),
        sessions=int(os.environ.get("RAG_HTTP_MAX_SESSIONS", defaults.sessions)),
        turns=int(os.environ.get("RAG_HTTP_MAX_TURNS", defaults.turns)),
        messages=int(os.environ.get("RAG_HTTP_MAX_MESSAGES", defaults.messages)),
        concurrent=int(os.environ.get("RAG_HTTP_MAX_CONCURRENT", defaults.concurrent)),
    )


async def _expire_idle(store):
    while True:
        await asyncio.sleep(min(store.limits.idle_seconds, 30))
        store.prune()


def create_app(*, runtime_loader=load_runtime, conversations=None):
    auth = InternalAuth.from_env()
    mode = os.environ.get("RAG_HTTP_MODE", "simulated")
    if mode not in MODES:
        raise ValueError("RAG_HTTP_MODE deve ser demonstrative, simulated ou semantic_real.")
    store = Conversations(_limits_from_env()) if conversations is None else conversations

    @asynccontextmanager
    async def lifespan(app):
        with ExitStack() as resources:
            try:
                app.state.runtime = resources.enter_context(runtime_loader(mode))
                app.state.unavailable = None
            except ApiError as error:
                app.state.runtime = None
                app.state.unavailable = error
            expiry = asyncio.create_task(_expire_idle(store))
            try:
                yield
            finally:
                expiry.cancel()
                with suppress(asyncio.CancelledError):
                    await expiry

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    bearer = HTTPBearer(auto_error=False)

    def subject(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        try:
            if credentials is not None:
                return auth.subject(credentials.credentials)
        except PermissionError:
            raise ApiError(403, "INSUFFICIENT_SCOPE", "Escopo interno insuficiente.") from None
        except (jwt.InvalidTokenError, OverflowError, ValueError, TypeError):
            pass
        raise ApiError(401, "INVALID_CREDENTIAL", "Credencial interna inválida ou ausente.")

    def runtime():
        error = app.state.unavailable
        if error is not None:
            raise ApiError(error.status, error.code, str(error), retryable=error.retryable)
        return app.state.runtime

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, error: ApiError):
        headers = {"WWW-Authenticate": "Bearer"} if error.status == 401 else None
        return JSONResponse(
            status_code=error.status, headers=headers,
            content={"error": {"code": error.code, "message": str(error), "retryable": error.retryable}},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, error: RequestValidationError):
        return await handle_api_error(request, ApiError(422, "INVALID_REQUEST", "Requisição inválida."))

    @app.get(PREFIX + "/health/live", dependencies=[Depends(subject)])
    def live():
        return {"status": "alive"}

    @app.get(PREFIX + "/health/ready", dependencies=[Depends(subject)])
    def ready():
        runtime()
        return {"status": "ready", "mode": mode}

    @app.post(PREFIX + "/conversations", status_code=201)
    def create(body: CreateConversation, owner: str = Depends(subject)) -> ConversationResponse:
        active = runtime()
        key, expiration = store.create(owner, lambda: active.session(store.limits.turns))
        return ConversationResponse(conversationId=key, mode=mode, expiresAt=expiration)

    @app.post(PREFIX + "/conversations/{conversationId}/messages")
    def message(conversationId: UUID, body: Question, owner: str = Depends(subject)) -> AnswerResponse:
        active = runtime()
        try:
            result = store.ask(conversationId, owner, body.messageId, body.question)
        except TimeoutError:
            raise ApiError(504, "PROVIDER_TIMEOUT", "O tutor excedeu o tempo de resposta.", retryable=True) from None
        except ProviderError:
            raise ApiError(503, "PROVIDER_UNAVAILABLE", "O provedor está temporariamente indisponível.",
                           retryable=True) from None
        except EmbeddingUnavailable:
            raise ApiError(503, "MODEL_UNAVAILABLE", "Modelo semântico temporariamente indisponível.",
                           retryable=True) from None
        except ChromaError:
            raise ApiError(503, "INDEX_UNAVAILABLE", "O índice do tutor está indisponível.",
                           retryable=True) from None
        evidence = []
        if result.evidence_id is not None:
            evidence.append(EvidenceResponse(id=result.evidence_id, corpus=active.corpus, corpusVersion=active.version))
        return AnswerResponse(
            conversationId=conversationId, messageId=body.messageId, mode=mode,
            outcome="answered" if result.answered else "no_evidence",
            answer=result.answer, evidence=evidence,
        )

    @app.delete(PREFIX + "/conversations/{conversationId}", status_code=204)
    def close(conversationId: UUID, owner: str = Depends(subject)):
        store.close(conversationId, owner)
        return Response(status_code=204)

    return app
