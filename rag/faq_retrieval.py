import re

from rag.session import Evidence


# A FAQ cobre o uso do aplicativo, sem emissão ou renovação de documentos oficiais.
_PASSPORT_SERVICE = re.compile(
    r"\b(?:emit(?:ir|o)|tir(?:ar|o)|renov(?:ar|o)|solicit(?:ar|o)|faz(?:er|o)|obter|obtenho|"
    r"emiss[aã]o|renovaç[aã]o|solicitaç[aã]o)\b[^.!?;]*\bpassaportes?\b",
    re.IGNORECASE,
)


class FaqRetriever:
    def __init__(self, collection, *, max_distance):
        if not 0 <= max_distance <= 2:
            raise ValueError("A distância cosseno máxima deve estar entre 0 e 2.")
        self.collection = collection
        self.max_distance = max_distance

    def retrieve(self, question):
        # Em continuações, o texto antes da quebra contém a pergunta que ancorou a sessão.
        if _PASSPORT_SERVICE.search(question.rsplit("\n", 1)[-1]):
            return []
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
