import argparse
import hashlib
import json
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings

from rag.semantic_embeddings import SemanticEmbedding

ROOT = Path(__file__).resolve().parents[1]
FAQ_FILE = ROOT / "knowledge" / "orienta_faq.json"
FAQ_DB = ROOT / "data" / "chroma_orienta"
METRIC = "cosine"
VERIFIED = "validado_por_inspecao_de_codigo"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_faq(path=FAQ_FILE):
    corpus = json.loads(path.read_text(encoding="utf-8"))
    version, rows = corpus["versao"], corpus["registros"]
    if type(version) is not int or version < 1 or not isinstance(rows, list) or not rows:
        raise ValueError("Informe uma versão inteira positiva e ao menos um registro.")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), str) and row[key].strip() for key in
            ("id", "categoria", "pergunta", "resposta", "evidencia", "escopo_verificacao")
        ):
            raise ValueError("Registro de FAQ incompleto.")
        identifier = row["id"]
        if not re.fullmatch(r"[A-Z][0-9]{2,}", identifier) or identifier in ids:
            raise ValueError(f"ID inválido ou duplicado: {identifier}")
        ids.add(identifier)
        sources = row.get("fontes")
        if row.get("status") != VERIFIED or not isinstance(sources, list) or not sources:
            raise ValueError(f"Registro sem verificação ou fontes: {identifier}")
        for source in sources:
            if (not isinstance(source, dict)
                    or not all(isinstance(source.get(key), str) and source[key].strip()
                        for key in ("repositorio", "arquivo", "simbolo", "commit"))
                    or not re.fullmatch(r"[0-9a-f]{40}", source["commit"])):
                raise ValueError(f"Fonte inválida: {identifier}")
    return version, sorted(rows, key=lambda row: row["id"])


def _manifest(version, rows, embedding):
    return {
        "schema": 1,
        "corpus_version": version,
        "corpus_sha256": hashlib.sha256(_canonical(rows).encode()).hexdigest(),
        "count": len(rows),
        "metric": METRIC,
        "embedding_function": embedding.name(),
        "embedding": embedding.get_config(),
    }


def _client(path):
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def _name(manifest):
    digest = hashlib.sha256(_canonical(manifest).encode()).hexdigest()[:16]
    return f"orienta_faq_v{manifest['corpus_version']}_{digest}"


def _validate(collection, embedding, *, manifest=None, require_ready=True):
    metadata = collection.metadata or {}
    stored = json.loads(metadata.get("manifest", "{}"))
    config = collection.configuration
    stored_embedding = config.get("embedding_function")
    if (
        stored.get("schema") != 1
        or type(stored.get("corpus_version")) is not int
        or stored.get("metric") != METRIC
        or stored.get("embedding_function") != embedding.name()
        or stored.get("embedding") != embedding.get_config()
        or (manifest is not None and stored != manifest)
        or _name(stored) != collection.name
        or (config.get("hnsw") or {}).get("space") != METRIC
        or stored_embedding is None
        or stored_embedding.name() != embedding.name()
        or stored_embedding.get_config() != embedding.get_config()
    ):
        raise ValueError("Coleção incompatível com o perfil de embeddings ou manifesto.")
    if require_ready:
        if metadata.get("ready") is not True or collection.count() != stored.get("count"):
            raise ValueError("Índice incompleto. Execute a indexação da FAQ.")
        vectors = collection.get(limit=1, include=["embeddings"])["embeddings"]
        if vectors is None or len(vectors) != 1 or len(vectors[0]) != stored["embedding"]["dimensions"]:
            raise ValueError("Dimensão armazenada incompatível com o manifesto.")
    return stored


def index_faq(path=FAQ_DB, faq_file=FAQ_FILE, *, embedding=None):
    embedding = SemanticEmbedding() if embedding is None else embedding
    version, rows = load_faq(faq_file)
    embedding.ensure_available()
    manifest = _manifest(version, rows, embedding)
    name = _name(manifest)
    client = _client(path)
    collection = client.get_or_create_collection(
        name=name, embedding_function=embedding,
        configuration={"hnsw": {"space": METRIC}},
        metadata={"manifest": _canonical(manifest), "ready": False},
    )
    _validate(collection, embedding, manifest=manifest, require_ready=False)
    if (collection.metadata or {}).get("ready") is not True:
        for offset in range(0, len(rows), 8):
            batch = rows[offset:offset + 8]
            collection.upsert(
                ids=[row["id"] for row in batch],
                documents=[f"{row['pergunta']}\n{row['resposta']}" for row in batch],
                metadatas=[{
                    "category": row["categoria"], "answer": row["resposta"],
                    "sources": _canonical(row["fontes"]), "status": row["status"],
                } for row in batch],
            )
        if collection.count() != manifest["count"]:
            raise ValueError("A contagem do índice não corresponde ao manifesto.")
        collection.modify(metadata={"manifest": _canonical(manifest), "ready": True})
    _validate(collection, embedding, manifest=manifest)
    return collection


def open_faq(path=FAQ_DB, faq_file=FAQ_FILE, *, name=None, embedding=None):
    embedding = SemanticEmbedding() if embedding is None else embedding
    manifest = None if name is not None else _manifest(*load_faq(faq_file), embedding)
    name = name if name is not None else _name(manifest)
    if not path.is_dir():
        raise ValueError("Índice Orienta ausente. Execute a indexação da FAQ.")
    collection = _client(path).get_collection(name, embedding_function=embedding)
    _validate(collection, embedding, manifest=manifest)
    return collection


def main():
    parser = argparse.ArgumentParser(description="Indexa a FAQ do Orienta com pesos locais.")
    parser.parse_args()
    try:
        collection = index_faq()
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    print(f"{collection.name}: {collection.count()} registros.")


if __name__ == "__main__":
    main()
