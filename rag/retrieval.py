from pathlib import Path

DEMO_DB = Path(__file__).resolve().parents[1] / "data" / "chroma"
COLLECTION_NAME = "faq_colecao"
MAX_DISTANCE = 0.72


def demo_collection(path=DEMO_DB, *, create=False):
    import chromadb
    from chromadb.config import Settings

    from rag.embeddings import PROFILE, DemoEmbedding

    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    if create:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=DemoEmbedding(),
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_profile": PROFILE},
        )
    else:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=DemoEmbedding())
    if (collection.metadata or {}).get("embedding_profile") != PROFILE:
        raise ValueError("A coleção não tem o perfil de embeddings da demonstração.")
    return collection


class ChromaRetriever:
    def __init__(self, collection):
        self.collection = collection

    def retrieve(self, question):
        result = self.collection.query(
            query_texts=[question], n_results=2, include=["documents", "distances"]
        )
        documents = (result["documents"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            document for document, distance in zip(documents, distances)
            if document and distance <= MAX_DISTANCE
        ]
