from chromadb import Documents, EmbeddingFunction
from chromadb.utils.embedding_functions import register_embedding_function

MODEL = "intfloat/multilingual-e5-small"
REVISION = "7cf867cae19b6df205046e9c11907797649fb552"


class EmbeddingUnavailable(RuntimeError):
    pass


@register_embedding_function
class SemanticEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, encoder=None):
        self._encoder = encoder

    @staticmethod
    def name():
        return "orienta-e5-small-v1"

    def get_config(self):
        return {
            "model": MODEL,
            "revision": REVISION,
            "dimensions": 384,
            "normalization": "l2",
            "document_prefix": "passage: ",
            "query_prefix": "query: ",
        }

    @staticmethod
    def build_from_config(config):
        embedding = SemanticEmbedding()
        if config != embedding.get_config():
            raise ValueError("Configuração de embeddings incompatível.")
        return embedding

    def ensure_available(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(
                    MODEL, revision=REVISION, device="cpu", local_files_only=True,
                    model_kwargs={"use_safetensors": True},
                )
            except (ImportError, OSError) as error:
                raise EmbeddingUnavailable(
                    "Instale as dependências semânticas e disponibilize no cache a revisão aprovada do modelo."
                ) from error

    def _encode(self, texts, prefix):
        self.ensure_available()
        try:
            vectors = self._encoder.encode(
                [prefix + text for text in texts],
                normalize_embeddings=True, batch_size=4, show_progress_bar=False,
            )
        except RuntimeError as error:
            raise EmbeddingUnavailable("A inferência semântica falhou.") from error
        if any(len(vector) != 384 for vector in vectors):
            raise EmbeddingUnavailable("O encoder deve produzir vetores de 384 dimensões.")
        return vectors

    def document_text(self, row):
        return f"{row['pergunta']}\n{row['resposta']}"

    def __call__(self, input: Documents):
        return self._encode(input, "passage: ")

    def embed_query(self, input: Documents):
        return self._encode(input, "query: ")


def configured_embedding():
    import os

    backend = os.environ.get("RAG_EMBEDDING_BACKEND", "sentence_transformers")
    if backend == "sentence_transformers":
        return SemanticEmbedding()
    if backend == "onnx":
        from rag.onnx_embeddings import OnnxE5Embedding
        return OnnxE5Embedding()
    raise ValueError("RAG_EMBEDDING_BACKEND deve ser sentence_transformers ou onnx.")
