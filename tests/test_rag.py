import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from examples.seed_demo import seed
from rag.cli import main
from rag.providers import ConfigurationError, GeminiProvider, OpenAIProvider, load_settings
from rag.retrieval import ChromaRetriever, demo_collection
from rag.session import NO_CONTEXT, Message, ProviderError, Session


class RetrievalTests(unittest.TestCase):
    def test_queries_two_and_filters_by_distance(self):
        collection = Mock()
        collection.query.return_value = {"documents": [["relevante", "distante"]], "distances": [[0.2, 0.95]]}
        self.assertEqual(ChromaRetriever(collection).retrieve("Pergunta"), ["relevante"])
        collection.query.assert_called_once_with(
            query_texts=["Pergunta"], n_results=2, include=["documents", "distances"]
        )

    def test_real_chroma_persistence_and_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chroma"
            seed(path)
            seed(path)
            collection = demo_collection(path)
            self.assertTrue((path / "chroma.sqlite3").is_file())
            self.assertEqual(collection.count(), 3)
            retriever = ChromaRetriever(collection)
            documents = retriever.retrieve("Reserva do Observatório Aurora")
            self.assertEqual(len(documents), 2)
            self.assertTrue(all("Observatório Aurora" in doc for doc in documents))
            self.assertEqual(retriever.retrieve("Xilofone submarino"), [])
            provider = Mock()
            provider.generate.return_value = "No portal da Academia Horizonte."
            session = Session(retriever, provider)
            self.assertEqual(session.ask("Reserva do Observatório Aurora"), provider.generate.return_value)
            self.assertIn("[2]", provider.generate.call_args.args[0][-1].content)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock()
        self.provider = Mock()
        self.provider.generate.return_value = "Vá ao portal."
        self.session = Session(self.retriever, self.provider)

    def test_context_and_history_order_without_duplicating_question(self):
        self.retriever.retrieve.return_value = ["Primeiro", "Segundo", "Terceiro"]
        self.session.ask("Onde reservo?")
        self.session.ask("E quando?")
        messages = self.provider.generate.call_args.args[0]
        self.assertEqual([message.role for message in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1].content, "Onde reservo?")
        self.assertIn("[2] Segundo", messages[-1].content)
        self.assertNotIn("Terceiro", messages[-1].content)
        self.assertEqual(self.session.history[-2], Message("user", "E quando?"))

    def test_elliptical_follow_up_retries_with_previous_question(self):
        self.retriever.retrieve.side_effect = [["Reserva"], [], ["Prazo"]]
        self.session.ask("Reserva do Observatório Aurora")
        self.session.ask("E quando?")
        self.retriever.retrieve.assert_called_with("Reserva do Observatório Aurora\nE quando?")
        self.assertIn("Prazo", self.provider.generate.call_args.args[0][-1].content)

    def test_no_context_skips_provider_and_records_turn(self):
        self.retriever.retrieve.side_effect = [["Reserva"], []]
        self.session.ask("Reserva")
        self.provider.generate.reset_mock()
        self.assertEqual(self.session.ask("Xilofone submarino"), NO_CONTEXT)
        self.provider.generate.assert_not_called()
        self.assertEqual(self.session.history[-1], Message("assistant", NO_CONTEXT))

    def test_failed_turn_can_be_retried_without_duplication(self):
        self.retriever.retrieve.return_value = ["Reserva"]
        self.provider.generate.side_effect = [ProviderError("Falha simulada"), "Portal"]
        with self.assertRaises(ProviderError):
            self.session.ask("Reserva?")
        self.assertEqual(len(self.session.history), 1)
        self.assertEqual(self.session.ask("Reserva?"), "Portal")
        self.assertEqual(len(self.session.history), 3)
        self.assertEqual(len(Session(self.retriever, self.provider).history), 1)


class ProviderTests(unittest.TestCase):
    def test_invalid_configuration(self):
        with self.assertRaisesRegex(ConfigurationError, "RAG_PROVIDER"):
            load_settings({"RAG_PROVIDER": "invalid"})
        with self.assertRaisesRegex(ConfigurationError, "GEMINI_MODEL"):
            load_settings({"GEMINI_API_KEY": "fixture", "GEMINI_MODEL": " "})

    def test_missing_keys_and_selected_provider_override(self):
        for provider in ("gemini", "openai"):
            with self.subTest(provider=provider), self.assertRaisesRegex(ConfigurationError, provider.upper()):
                load_settings({}, selected=provider)
        self.assertEqual(load_settings({"OPENAI_API_KEY": "fixture"}, selected="openai").model, "gpt-4o-mini")
        self.assertEqual(load_settings({"GEMINI_API_KEY": "fixture"}).provider, "gemini")

    def test_switch_preserves_retrieval_and_maps_roles(self):
        retriever = Mock()
        retriever.retrieve.return_value = ["Evidência"]
        gemini, openai = Mock(), Mock()
        gemini.models.generate_content.return_value = SimpleNamespace(text="Resposta")
        openai.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Outra resposta"))]
        )
        session = Session(retriever, GeminiProvider(gemini, "gemini-fixture"))
        session.ask("Primeira?")
        session.ask("Segunda?")
        gemini_args = gemini.models.generate_content.call_args.kwargs
        self.assertEqual([item["role"] for item in gemini_args["contents"]], ["user", "model", "user"])
        self.assertIn("tutor", gemini_args["config"]["system_instruction"])
        session.provider = OpenAIProvider(openai, "gpt-4o-mini")
        session.ask("Terceira?")
        args = openai.chat.completions.create.call_args.kwargs
        self.assertEqual(args["temperature"], 0)
        self.assertEqual([item["role"] for item in args["messages"]],
                         ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertIn("Evidência", args["messages"][-1]["content"])
        self.assertEqual(retriever.retrieve.call_count, 3)

    def test_provider_errors_are_sanitized(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("segredo simulado")
        client.models.generate_content.side_effect = RuntimeError("segredo simulado")
        for provider in (OpenAIProvider(client, "fixture"), GeminiProvider(client, "fixture")):
            with self.subTest(provider=type(provider).__name__), self.assertRaisesRegex(ProviderError, "^Falha na geração"):
                provider.generate((Message("user", "Pergunta"),))

    def test_cli_requires_opt_in(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as outcome:
            main([])
        self.assertEqual(outcome.exception.code, 2)

    def test_cli_processes_turn_and_exits_using_simulated_provider(self):
        output = StringIO()
        collection = Mock()
        collection.query.return_value = {"documents": [["Portal"]], "distances": [[0.2]]}
        with (
            redirect_stdout(output),
            patch("dotenv.load_dotenv"),
            patch("rag.cli.load_settings") as settings,
            patch("rag.cli.demo_collection", return_value=collection),
            patch("rag.cli.make_provider") as provider,
            patch("builtins.input", side_effect=["Onde reservo?", "/sair"]),
        ):
            settings.return_value.provider = "gemini"
            provider.return_value.generate.return_value = "No portal."
            main(["--allow-api"])
        self.assertIn("Tutor: No portal.", output.getvalue())
        provider.return_value.generate.assert_called_once()
