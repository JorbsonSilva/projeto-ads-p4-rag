import hashlib
import re
import unicodedata

from chromadb import Documents, EmbeddingFunction
from chromadb.utils.embedding_functions import register_embedding_function

PROFILE = "orienta-demo-lexical-v1"
STOP_WORDS = frozenset(
    "como para por qual quais uma que dos das com isso sobre".split()
)


@register_embedding_function
class DemoEmbedding(EmbeddingFunction[Documents]):
    """Vetores lexicais locais e reproduzíveis para o corpus fictício."""

    def __init__(self, dimensions=2048):
        self.dimensions = dimensions

    def __call__(self, input: Documents):
        vectors = []
        for text in input:
            normalized = unicodedata.normalize("NFKD", text.casefold())
            normalized = "".join(char for char in normalized if not unicodedata.combining(char))
            vector = [0.0] * self.dimensions
            tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", normalized)) - STOP_WORDS
            for token in tokens:
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = 1 + int.from_bytes(digest, "big") % (len(vector) - 1)
                vector[index] += 1.0
            if not tokens:
                vector[0] = 1.0
            vectors.append(vector)
        return vectors

    @staticmethod
    def name():
        return PROFILE

    def get_config(self):
        return {"dimensions": self.dimensions}

    @staticmethod
    def build_from_config(config):
        return DemoEmbedding(config["dimensions"])
