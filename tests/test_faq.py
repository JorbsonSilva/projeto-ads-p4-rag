import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from rag.faq_retrieval import FaqRetriever
from rag.index_faq import FAQ_FILE, index_faq, load_faq, open_faq
from rag.providers import GeminiProvider, OpenAIProvider
from rag.semantic_embeddings import SemanticEmbedding
from rag.session import NO_CONTEXT, ProviderError, Session


class FixtureEncoder:
    """Define vetores de teste por intenção, sem medir a semântica do modelo real."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.extend((text, kwargs) for text in texts)
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            text = text.casefold()
            if "remarc" in text or "trocar o horário" in text:
                vector[0], vector[1] = 1.0, 0.2
            elif "cancelar" in text or "desistir" in text:
                vector[0], vector[1] = 0.2, 1.0
            else:
                vector[2 if text.startswith("query:") else 3] = 1.0
            vectors.append(vector)
        return vectors


class FaqTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.db = self.root / "chroma"
        self.encoder = FixtureEncoder()
        self.embedding = SemanticEmbedding(encoder=self.encoder)
        self.collection = index_faq(self.db, embedding=self.embedding)
        self.retriever = FaqRetriever(self.collection, max_distance=0.15)

    def test_real_chroma_idempotence_and_reopen_with_simulated_encoder(self):
        repeat = index_faq(self.db, embedding=self.embedding)
        reopened = open_faq(self.db, embedding=self.embedding)
        self.assertEqual(reopened.name, repeat.name)
        self.assertEqual(reopened.count(), 33)
        self.assertEqual(len(set(reopened.get(include=[])["ids"])), 33)
        manifest = json.loads(reopened.metadata["manifest"])
        self.assertEqual(manifest["embedding"]["dimensions"], 384)
        self.assertEqual(manifest["metric"], "cosine")
        self.assertEqual(manifest["corpus_version"], 1)
        row = reopened.get(ids=["M07"], include=["metadatas"])["metadatas"][0]
        self.assertEqual(row["category"], "Mentorias")
        self.assertIn("remarcarAgendamento", row["sources"])
        self.assertTrue((self.db / "chroma.sqlite3").is_file())

    def test_update_creates_revision_with_same_ids_and_old_version_can_be_reopened(self):
        faq = json.loads(FAQ_FILE.read_text(encoding="utf-8"))
        faq["versao"] += 1
        row = next(row for row in faq["registros"] if row["id"] == "M07")
        row["resposta"] += " Atualização simulada."
        updated = self.root / "updated.json"
        updated.write_text(json.dumps(faq), encoding="utf-8")
        latest = index_faq(self.db, updated, embedding=self.embedding)
        previous = open_faq(self.db, name=self.collection.name, embedding=self.embedding)
        self.assertNotEqual(latest.name, previous.name)
        self.assertEqual(set(latest.get(include=[])["ids"]), set(previous.get(include=[])["ids"]))
        self.assertIn("Atualização simulada.", latest.get(ids=["M07"])["documents"][0])
        self.assertNotIn("Atualização simulada.", previous.get(ids=["M07"])["documents"][0])

    def test_duplicate_ids_are_rejected_before_indexing(self):
        faq = json.loads(FAQ_FILE.read_text(encoding="utf-8"))
        faq["registros"].append(faq["registros"][0])
        duplicate = self.root / "duplicate.json"
        duplicate.write_text(json.dumps(faq), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicado"):
            load_faq(duplicate)

    def test_simulated_paraphrases_distinguish_cancel_and_reschedule(self):
        for question, expected in [
            ("Consigo trocar o horário da mentoria?", "M07"),
            ("Quero desistir da aula, até quando posso?", "M08"),
        ]:
            with self.subTest(question=question):
                self.assertEqual([r.identifier for r in self.retriever.retrieve(question)], [expected])
        self.assertTrue(any(text.startswith("passage: ") for text, _ in self.encoder.calls))
        self.assertTrue(any(text.startswith("query: ") for text, _ in self.encoder.calls))
        self.assertTrue(all(options["normalize_embeddings"] for _, options in self.encoder.calls))

    def test_out_of_scope_and_invented_provider_text(self):
        provider = Mock()
        session = Session(self.retriever, provider, operational_faq=True)
        self.assertEqual(session.ask("Qual é a previsão do tempo?"), NO_CONTEXT)
        provider.generate.assert_not_called()
        provider.generate.return_value = "M07. Agendei sua aula para amanhã."
        self.assertEqual(session.ask("Consigo trocar o horário?"), NO_CONTEXT)
        provider.generate.return_value = "SEM_EVIDENCIA"
        self.assertEqual(session.ask("Como remarco e quem paga o estorno?"), NO_CONTEXT)
        provider.generate.return_value = "M08"
        self.assertEqual(session.ask("Como remarco?"), NO_CONTEXT)

    def test_history_isolation_and_provider_failure(self):
        provider = Mock()
        session = Session(self.retriever, provider, operational_faq=True)
        other = Session(self.retriever, provider, operational_faq=True)
        provider.generate.side_effect = [ProviderError("Falha"), "M07", "M08"]
        with self.assertRaises(ProviderError):
            session.ask("Como remarco?")
        self.assertEqual(len(session.history), 1)
        self.assertIn("Fonte: M07", session.ask("Como remarco?"))
        self.assertIn("Fonte: M08", session.ask("E se eu quiser cancelar?"))
        self.assertEqual([item.role for item in session.history],
                         ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual(len(other.history), 1)

    def test_provider_adapters_select_the_same_evidence(self):
        gemini, openai = Mock(), Mock()
        gemini.models.generate_content.return_value = SimpleNamespace(text="M07")
        openai.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="M07"))],
        )
        question = "Consigo trocar o horário?"
        first = Session(self.retriever, GeminiProvider(gemini, "fixture"), operational_faq=True)
        second = Session(self.retriever, OpenAIProvider(openai, "fixture"), operational_faq=True)
        self.assertEqual(first.ask(question), second.ask(question))
        self.assertIn("[M07]", gemini.models.generate_content.call_args.kwargs["contents"][-1]["parts"][0]["text"])
        self.assertIn("[M07]", openai.chat.completions.create.call_args.kwargs["messages"][-1]["content"])

    def test_wrong_model_profile_is_rejected_before_query(self):
        class WrongProfile(SemanticEmbedding):
            def get_config(self):
                return dict(super().get_config(), revision="outra")

        with self.assertRaisesRegex(ValueError, "incompatível"):
            open_faq(self.db, name=self.collection.name, embedding=WrongProfile(self.encoder))
