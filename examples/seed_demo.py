from rag.retrieval import DEMO_DB, demo_collection

DOCUMENTS = {
    "observatorio-reserva": (
        "Exemplo fictício. A reserva do Observatório Aurora é feita no portal "
        "da Academia Horizonte, na seção Laboratórios."
    ),
    "observatorio-prazo": (
        "Exemplo fictício. A reserva do Observatório Aurora abre na segunda-feira "
        "e permite cancelar até 24 horas antes da visita."
    ),
    "biblioteca-emprestimo": (
        "Exemplo fictício. A Biblioteca Nébula empresta livros por sete dias "
        "e permite renovação pelo catálogo digital."
    ),
}


def seed(path=DEMO_DB):
    collection = demo_collection(path, create=True)
    collection.upsert(ids=list(DOCUMENTS), documents=list(DOCUMENTS.values()))
    return collection


if __name__ == "__main__":
    print(f"Base demonstrativa pronta com {seed().count()} trechos.")
