"""Suíte opcional. Requer dependências e o checkpoint já disponível localmente."""

import tempfile
import unittest
from pathlib import Path

from rag.faq_retrieval import FaqRetriever
from rag.index_faq import index_faq, open_faq
from rag.semantic_embeddings import SemanticEmbedding


class RealModelTests(unittest.TestCase):
    def test_portuguese_paraphrases_in_top_two(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            embedding = SemanticEmbedding()
            index_faq(path, embedding=embedding)
            collection = open_faq(path, embedding=embedding)
            # Aqui só se mede ranking. O limiar operacional exige calibração com negativos.
            retriever = FaqRetriever(collection, max_distance=2)
            examples = {
                "Consigo trocar o horário da mentoria?": "M07",
                "Quero desistir da aula, até quando posso?": "M08",
                "Qual documento posso usar para entrar na conta?": "C03",
                "Dá para mandar uma imagem na conversa da aula?": "H02",
            }
            for question, expected in examples.items():
                with self.subTest(question=question):
                    ids = [record.identifier for record in retriever.retrieve(question)]
                    self.assertIn(expected, ids)
