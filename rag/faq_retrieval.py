from rag.session import Evidence


class FaqRetriever:
    def __init__(self, collection, *, max_distance):
        if not 0 <= max_distance <= 2:
            raise ValueError("A distância cosseno máxima deve estar entre 0 e 2.")
        self.collection = collection
        self.max_distance = max_distance

    def retrieve(self, question):
        result = self.collection.query(
            query_texts=[question], n_results=2, include=["documents", "distances", "metadatas"]
        )
        return [
            Evidence(identifier, document, metadata["answer"])
            for identifier, document, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0],
                result["distances"][0], strict=True,
            )
            if distance <= self.max_distance
        ]
