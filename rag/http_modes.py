import os
import re
import shutil
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from chromadb.errors import ChromaError

from rag.http_sessions import ApiError
from rag.retrieval import DEMO_DB, ChromaRetriever, demo_collection
from rag.session import Evidence, Generator, Retriever, Session

Mode = Literal["demonstrative", "simulated", "semantic_real"]
MODES = ("demonstrative", "simulated", "semantic_real")


@dataclass(frozen=True)
class Runtime:
    retriever: Retriever
    provider: Generator
    corpus: str
    version: int
    generative: bool = False

    def session(self, max_turns):
        return Session(self.retriever, self.provider, operational_faq=True, max_turns=max_turns,
                       generative=self.generative)


class LocalSelector:
    def generate(self, messages):
        candidate = re.match(r"Trechos recuperados:\n\[([^\]]+)\]", messages[-1].content)
        return candidate[1] if candidate else "SEM_EVIDENCIA"


def _normalize(text):
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


class SimulatedFaq:
    def __init__(self, rows):
        records = {
            row["id"]: Evidence(row["id"], f"{row['pergunta']}\n{row['resposta']}", row["resposta"])
            for row in rows
        }
        self.questions = {_normalize(row["pergunta"]): records[row["id"]] for row in rows}
        self.questions.update({
            _normalize("Consigo trocar o horário da mentoria?"): records["M07"],
            _normalize("Quero desistir da aula. Até quando posso?"): records["M08"],
        })

    def retrieve(self, question):
        previous, separator, followup = question.partition("\n")
        if separator and _normalize(followup) not in {"e quando", "e como"}:
            return []
        evidence = self.questions.get(_normalize(previous))
        return [evidence] if evidence else []


class DemoEvidence:
    def __init__(self, retriever):
        from examples.seed_demo import DOCUMENTS

        self.retriever = retriever
        self.ids = {document: identifier for identifier, document in DOCUMENTS.items()}

    def retrieve(self, question):
        return [Evidence(self.ids[text], text, text)
                for text in self.retriever.retrieve(question) if text in self.ids]


@contextmanager
def load_runtime(mode):
    if mode == "simulated":
        from rag.index_faq import load_faq

        version, rows = load_faq()
        yield Runtime(SimulatedFaq(rows), LocalSelector(), "orienta_faq", version)
        return

    if mode == "semantic_real":
        from dotenv import load_dotenv

        from rag.faq_retrieval import FaqRetriever
        from rag.index_faq import FAQ_DB, ROOT, load_faq, open_faq
        from rag.providers import load_settings, make_provider
        from rag.semantic_embeddings import configured_embedding

        load_dotenv(ROOT / ".env")
        if not FAQ_DB.is_dir():
            raise ApiError(503, "INDEX_UNAVAILABLE", "Indexe a FAQ antes de iniciar o Tutor.", retryable=True)
        try:
            embedding = configured_embedding()
            collection = open_faq(FAQ_DB, embedding=embedding)
            version, _ = load_faq()
            retriever = FaqRetriever(collection, max_distance=float(os.environ["RAG_MAX_DISTANCE"]))
        except (ChromaError, OSError, KeyError, ValueError, RuntimeError):
            raise ApiError(503, "INDEX_INCOMPATIBLE",
                           "Verifique o índice, o perfil de embeddings e RAG_MAX_DISTANCE.",
                           retryable=True) from None
        try:
            embedding.ensure_available()
        except RuntimeError:
            raise ApiError(503, "MODEL_UNAVAILABLE", "Modelo semântico local indisponível.",
                           retryable=True) from None
        try:
            provider = make_provider(load_settings(), structured=True)
        except Exception:
            raise ApiError(503, "PROVIDER_UNAVAILABLE", "Configure o provedor de geração.",
                           retryable=True) from None
        try:
            yield Runtime(retriever, provider, "orienta_faq", version, generative=True)
        finally:
            provider.client.close()
        return

    if not (DEMO_DB / "chroma.sqlite3").is_file():
        raise ApiError(503, "INDEX_UNAVAILABLE", "A coleção demonstrativa está indisponível.", retryable=True)
    with TemporaryDirectory(prefix="orienta-http-demo-") as directory:
        try:
            copy = Path(directory) / "chroma"
            shutil.copytree(DEMO_DB, copy)
            retriever = DemoEvidence(ChromaRetriever(demo_collection(copy)))
        except (ChromaError, OSError, ValueError, RuntimeError) as error:
            raise ApiError(503, "INDEX_INCOMPATIBLE", "A coleção demonstrativa está incompatível.",
                           retryable=True) from error
        yield Runtime(retriever, LocalSelector(), "orienta_demo", 1)
