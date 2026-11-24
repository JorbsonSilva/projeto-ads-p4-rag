import hashlib
import os
import secrets
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch
from uuid import uuid4

import jwt
from fastapi.testclient import TestClient

from examples.seed_demo import seed
from rag.http_api import PREFIX, create_app
from rag.http_modes import Runtime, SimulatedFaq
from rag.http_sessions import Conversations, Limits
from rag.index_faq import load_faq
from rag.session import ProviderError, ProviderTimeout

CONVERSATIONS = PREFIX + "/conversations"
READY = PREFIX + "/health/ready"
LIVE = PREFIX + "/health/live"
QUESTION = "Consigo trocar o horário da mentoria?"


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.secret = secrets.token_urlsafe(48)
        environment = patch.dict(os.environ, {
            "RAG_INTERNAL_JWT_SECRET": self.secret,
            "RAG_INTERNAL_JWT_ISSUER": "spring-test",
            "RAG_INTERNAL_JWT_AUDIENCE": "rag-test",
            "RAG_HTTP_MODE": "simulated",
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def headers(self, sub="alice", *, signing_key=None, **claims):
        now = int(time.time())
        payload = {
            "iss": "spring-test", "aud": "rag-test", "sub": sub,
            "iat": now, "exp": now + 60, "scope": "rag:internal",
        } | claims
        return {"Authorization": "Bearer " + jwt.encode(payload, signing_key or self.secret, algorithm="HS256")}

    def create(self, client, owner="alice"):
        response = client.post(CONVERSATIONS, headers=self.headers(owner), json={})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertIn("expiresAt", response.json())
        return response.json()["conversationId"]

    def message(self, client, conversation, question=QUESTION, *, owner="alice", message_id=None):
        return client.post(
            f"{CONVERSATIONS}/{conversation}/messages", headers=self.headers(owner),
            json={"messageId": message_id or str(uuid4()), "question": question},
        )

    @contextmanager
    def runtime(self, provider):
        version, rows = load_faq()
        yield Runtime(SimulatedFaq(rows), provider, "orienta_faq", version)

    def test_missing_secret_fails_closed(self):
        os.environ.pop("RAG_INTERNAL_JWT_SECRET")
        with self.assertRaisesRegex(ValueError, "JWT interno"):
            create_app()

    def test_authentication_checks_signature_audience_issuer_expiry_scope_and_lifetime(self):
        now = int(time.time())
        invalid = [
            {}, self.headers(signing_key=secrets.token_urlsafe(48)),
            self.headers(aud="mobile"), self.headers(iss="mobile"), self.headers(exp=now-1),
            self.headers(exp=now+121), self.headers(iat=now+60), self.headers(sub=" "),
        ]
        with TestClient(create_app()) as client:
            for headers in invalid:
                with self.subTest(headers_present=bool(headers)):
                    response = client.post(CONVERSATIONS, headers=headers, json={})
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(response.json()["error"]["code"], "INVALID_CREDENTIAL")
            denied = client.get(READY, headers=self.headers(scope="rag:other"))
            self.assertEqual(denied.status_code, 403)
            self.assertEqual(client.get(LIVE).status_code, 401)

    def test_simulated_answer_no_evidence_and_close(self):
        with TestClient(create_app()) as client:
            self.assertEqual(client.get(LIVE, headers=self.headers()).json(), {"status": "alive"})
            self.assertEqual(client.get(READY, headers=self.headers()).json()["mode"], "simulated")
            conversation = self.create(client)
            answer = self.message(client, conversation).json()
            self.assertEqual(answer["outcome"], "answered")
            self.assertEqual(answer["mode"], "simulated")
            self.assertIn("24 horas", answer["answer"])
            self.assertEqual(answer["evidence"], [{"id": "M07", "corpus": "orienta_faq", "corpusVersion": 1}])
            empty = self.message(client, conversation, "Quem paga o estorno?").json()
            self.assertEqual(empty["outcome"], "no_evidence")
            self.assertEqual(empty["evidence"], [])
            closed = client.delete(f"{CONVERSATIONS}/{conversation}", headers=self.headers())
            self.assertEqual((closed.status_code, closed.content), (204, b""))
            self.assertEqual(self.message(client, conversation).status_code, 404)

    def test_validation_rejects_identity_mode_provider_and_invalid_question(self):
        with TestClient(create_app()) as client:
            conversation = self.create(client)
            for payload in [
                {"question": " ", "messageId": str(uuid4())},
                {"question": 42, "messageId": str(uuid4())},
                {"question": "a"*1001, "messageId": str(uuid4())},
                {"question": QUESTION, "messageId": "invalid"},
                *[{"question": QUESTION, "messageId": str(uuid4()), field: "tentativa"}
                  for field in ("usuarioId", "mode", "provider")],
            ]:
                response = client.post(f"{CONVERSATIONS}/{conversation}/messages", headers=self.headers(), json=payload)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")
            self.assertEqual(
                client.post(CONVERSATIONS, headers=self.headers(), json={"usuarioId": 5}).status_code, 422,
            )

    def test_isolation_and_expiration_are_indistinguishable_from_missing(self):
        elapsed = [0]
        store = Conversations(Limits(idle_seconds=5), clock=lambda: elapsed[0])
        with TestClient(create_app(conversations=store)) as client:
            conversation = self.create(client)
            missing = self.message(client, str(uuid4()))
            foreign = self.message(client, conversation, owner="bob")
            self.assertEqual((foreign.status_code, foreign.json()), (404, missing.json()))
            self.assertEqual(
                client.delete(f"{CONVERSATIONS}/{conversation}", headers=self.headers("bob")).json(),
                missing.json(),
            )
            elapsed[0] = 4
            self.assertEqual(self.message(client, conversation).status_code, 200)
            elapsed[0] = 8
            self.assertEqual(self.message(client, conversation).status_code, 200)
            elapsed[0] = 13
            expired = self.message(client, conversation)
            self.assertEqual((expired.status_code, expired.json()), (404, missing.json()))

    def test_idempotency_history_window_and_session_limit(self):
        provider = Mock()
        provider.generate.return_value = "M07"
        store = Conversations(Limits(sessions=2, turns=1, messages=3))
        with TestClient(create_app(runtime_loader=lambda mode: self.runtime(provider), conversations=store)) as client:
            one = self.create(client)
            two = self.create(client, "bob")
            self.assertEqual(client.post(CONVERSATIONS, headers=self.headers(), json={}).status_code, 429)
            message_id = str(uuid4())
            initial = self.message(client, one, message_id=message_id)
            repeated = self.message(client, one, "  " + QUESTION + " ", message_id=message_id)
            self.assertEqual(initial.json(), repeated.json())
            provider.generate.assert_called_once()
            clash = self.message(client, one, "Outra pergunta", message_id=message_id)
            self.assertEqual((clash.status_code, clash.json()["error"]["code"]), (409, "MESSAGE_CONFLICT"))
            self.assertEqual(self.message(client, one, "E quando?").json()["outcome"], "answered")
            messages = provider.generate.call_args.args[0]
            self.assertEqual([m.role for m in messages], ["system", "user", "assistant", "user"])
            self.assertEqual(messages[1].content, QUESTION)
            self.message(client, one)
            self.assertEqual(provider.generate.call_args.args[0][1].content, "E quando?")
            self.assertEqual(self.message(client, one).status_code, 429)
            self.message(client, two, owner="bob")
            self.assertEqual([m.role for m in provider.generate.call_args.args[0]], ["system", "user"])

    def test_concurrency_limits_and_busy_close(self):
        entered, release = Event(), Event()
        provider = Mock()

        def blocking(_messages):
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("O teste não liberou o turno.")
            return "M07"

        provider.generate.side_effect = blocking
        store = Conversations(Limits(concurrent=1))
        with TestClient(create_app(runtime_loader=lambda mode: self.runtime(provider), conversations=store)) as client:
            one, two = self.create(client), self.create(client)
            with ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(self.message, client, one)
                try:
                    self.assertTrue(entered.wait(timeout=3))
                    busy = self.message(client, one)
                    self.assertEqual((busy.status_code, busy.json()["error"]["code"]), (409, "TURN_IN_PROGRESS"))
                    over = self.message(client, two)
                    self.assertEqual((over.status_code, over.json()["error"]["code"]), (429, "CONCURRENCY_LIMIT"))
                    self.assertEqual(client.delete(f"{CONVERSATIONS}/{one}", headers=self.headers()).status_code, 409)
                finally:
                    release.set()
                self.assertEqual(pending.result(timeout=5).status_code, 200)

    def test_provider_errors_allow_retry_without_inserting_turn(self):
        provider = Mock()
        provider.generate.side_effect = [ProviderTimeout("interno"), ProviderError("interno"), "M07"]
        with TestClient(create_app(runtime_loader=lambda mode: self.runtime(provider))) as client:
            conversation = self.create(client)
            message_id = str(uuid4())
            for status, code in [(504, "PROVIDER_TIMEOUT"), (503, "PROVIDER_UNAVAILABLE")]:
                failure = self.message(client, conversation, message_id=message_id)
                self.assertEqual((failure.status_code, failure.json()["error"]["code"]), (status, code))
                self.assertNotIn("interno", failure.text)
            self.assertEqual(self.message(client, conversation, message_id=message_id).status_code, 200)
            self.assertEqual([m.role for m in provider.generate.call_args.args[0]], ["system", "user"])

    def test_demonstrative_uses_copy_and_returns_fictitious_source(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "original"
            seed(database)
            files = {path: hashlib.sha256(path.read_bytes()).digest()
                     for path in database.rglob("*") if path.is_file()}
            with (
                patch.dict(os.environ, {"RAG_HTTP_MODE": "demonstrative"}),
                patch("rag.http_modes.DEMO_DB", database),
                TestClient(create_app()) as client,
            ):
                conversation = self.create(client)
                response = self.message(client, conversation, "Reserva do Observatório Aurora")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["mode"], "demonstrative")
                self.assertEqual(data["outcome"], "answered")
                self.assertEqual(data["evidence"][0]["corpus"], "orienta_demo")
            self.assertEqual(files, {path: hashlib.sha256(path.read_bytes()).digest()
                                     for path in database.rglob("*") if path.is_file()})

    def test_semantic_mode_reports_missing_and_incompatible_index_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"RAG_HTTP_MODE": "semantic_real"}):
            absent = Path(directory) / "absent"
            with patch("rag.index_faq.FAQ_DB", absent), TestClient(create_app()) as client:
                self.assertEqual(client.get(LIVE, headers=self.headers()).status_code, 200)
                unavailable = client.get(READY, headers=self.headers())
                self.assertEqual((unavailable.status_code, unavailable.json()["error"]["code"]),
                                 (503, "INDEX_UNAVAILABLE"))
                self.assertEqual(client.post(CONVERSATIONS, headers=self.headers(), json={}).status_code, 503)
            incompatible = Path(directory) / "incompatible"
            incompatible.mkdir()
            with patch("rag.index_faq.FAQ_DB", incompatible), TestClient(create_app()) as client:
                unavailable = client.get(READY, headers=self.headers())
                self.assertEqual((unavailable.status_code, unavailable.json()["error"]["code"]),
                                 (503, "INDEX_INCOMPATIBLE"))

    def test_semantic_runtime_uses_retrieved_evidence_and_isolated_history(self):
        from rag.index_faq import index_faq
        from rag.semantic_embeddings import SemanticEmbedding
        from tests.test_faq import FixtureEncoder

        provider = Mock()
        provider.generate.return_value = '{"id":"M08","resposta":"Consulte o prazo de cancelamento da mentoria."}'
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "index"
            embedding = SemanticEmbedding(encoder=FixtureEncoder())
            index_faq(database, embedding=embedding)
            with (
                patch.dict(os.environ, {"RAG_HTTP_MODE": "semantic_real", "RAG_MAX_DISTANCE": "0.15"}),
                patch("rag.index_faq.FAQ_DB", database),
                patch("rag.semantic_embeddings.configured_embedding", return_value=embedding),
                patch("rag.providers.load_settings"),
                patch("rag.providers.make_provider", return_value=provider),
                TestClient(create_app()) as client,
            ):
                self.assertEqual(client.get(READY, headers=self.headers()).json()["mode"], "semantic_real")
                first, second = self.create(client), self.create(client)
                initial = self.message(client, first, "Como cancelar uma mentoria?").json()
                self.assertEqual(initial["mode"], "semantic_real")
                self.assertEqual(initial["answer"], "Consulte o prazo de cancelamento da mentoria.")
                self.assertEqual(initial["evidence"][0]["id"], "M08")
                followup = self.message(client, first, "E qual é o prazo?").json()
                self.assertEqual(followup["evidence"][0]["id"], "M08")
                history, = provider.generate.call_args.args
                self.assertEqual([m.role for m in history], ["system", "user", "assistant", "user"])
                isolated = self.message(client, second, "E qual é o prazo?").json()
                self.assertEqual(isolated["outcome"], "no_evidence")
                self.assertEqual(isolated["evidence"], [])
                self.assertEqual(provider.generate.call_count, 2)
                provider.generate.return_value = '{"id":"M07","resposta":"Resposta de outra regra."}'
                fabricated = self.message(client, first, "Como cancelar uma mentoria?").json()
                self.assertEqual(fabricated["evidence"], [])
                provider.generate.side_effect = ProviderTimeout("limite")
                response = self.message(client, first, "Como cancelar uma mentoria?")
                self.assertEqual((response.status_code, response.json()["error"]["code"]), (504, "PROVIDER_TIMEOUT"))
