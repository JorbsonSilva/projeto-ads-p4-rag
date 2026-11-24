import os
from pathlib import Path
from threading import Lock

from chromadb.utils.embedding_functions import register_embedding_function

from rag.semantic_embeddings import EmbeddingUnavailable, SemanticEmbedding

MODEL = "Xenova/multilingual-e5-small"
REVISION = "761b726dd34fb83930e26aab4e9ac3899aa1fa78"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "e5-onnx"
MAX_LENGTH = 512


@register_embedding_function
class OnnxE5Embedding(SemanticEmbedding):
    def __init__(self):
        super().__init__()
        self._lock = Lock()

    @staticmethod
    def name():
        return "orienta-e5-small-onnx-uint8-v1"

    def get_config(self):
        return super().get_config() | {
            "model": MODEL, "revision": REVISION, "artifact": "model_uint8.onnx",
            "pooling": "attention_mask_mean", "max_length": MAX_LENGTH,
            "document_fields": "pergunta resposta categoria",
        }

    @staticmethod
    def build_from_config(config):
        embedding = OnnxE5Embedding()
        if config != embedding.get_config():
            raise ValueError("Configuração ONNX incompatível.")
        return embedding

    def document_text(self, row):
        return " ".join(row[key] for key in ("pergunta", "resposta", "categoria"))

    def ensure_available(self):
        if self._encoder is not None:
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            directory = Path(os.environ.get("RAG_ONNX_MODEL_DIR", DEFAULT_PATH))
            tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
            tokenizer.enable_truncation(max_length=MAX_LENGTH)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.add_session_config_entry("session.disable_prepacking", "1")
            session = ort.InferenceSession(str(directory / "model_uint8.onnx"),
                                           sess_options=options, providers=["CPUExecutionProvider"])
            self._encoder = (tokenizer, session)
        except Exception as error:
            raise EmbeddingUnavailable("Modelo ONNX local indisponível ou incompatível.") from error

    def _encode(self, texts, prefix):
        import numpy as np

        with self._lock:
            self.ensure_available()
            tokenizer, session = self._encoder
            vectors = []
            for text in texts:
                tokens = tokenizer.encode(prefix + text)
                mask = np.asarray([tokens.attention_mask], dtype=np.int64)
                inputs = {
                    "input_ids": np.asarray([tokens.ids], dtype=np.int64),
                    "attention_mask": mask,
                    "token_type_ids": np.asarray([tokens.type_ids], dtype=np.int64),
                }
                try:
                    output, = session.run(["last_hidden_state"], {
                        entry.name: inputs[entry.name] for entry in session.get_inputs()
                    })
                except Exception as error:
                    raise EmbeddingUnavailable("A inferência ONNX falhou.") from error
                pooled = (output * mask[..., None]).sum(axis=1)[0] / mask.sum()
                norm = np.linalg.norm(pooled)
                if pooled.shape != (384,) or not np.isfinite(norm) or norm == 0:
                    raise EmbeddingUnavailable("Vetor ONNX inválido.")
                vectors.append((pooled / norm).astype(np.float32))
            return vectors
