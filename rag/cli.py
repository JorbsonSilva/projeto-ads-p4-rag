import argparse
from pathlib import Path

from rag.providers import ConfigurationError, load_settings, make_provider
from rag.retrieval import ChromaRetriever, demo_collection
from rag.session import NO_CONTEXT, ProviderError, Session


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tutor RAG do Orienta")
    parser.add_argument("--provider", choices=("gemini", "openai"))
    parser.add_argument("--corpus", choices=("demo", "orienta"), default="demo")
    parser.add_argument("--collection", help="Nome de uma versão anterior da coleção Orienta")
    parser.add_argument("--max-distance", type=float, help="Limiar cosseno da coleção Orienta")
    parser.add_argument("--retrieve-only", metavar="PERGUNTA", help="Consulte a FAQ local sem provedor")
    parser.add_argument("--allow-api", action="store_true", help="Autoriza envio de texto ao provedor selecionado")
    args = parser.parse_args(argv)
    operational = args.corpus == "orienta"
    if not operational and (args.collection or args.max_distance is not None or args.retrieve_only is not None):
        parser.error("Essas opções exigem --corpus orienta.")
    if operational and args.max_distance is None:
        parser.error("Informe --max-distance com um limiar de relevância para a FAQ.")
    if args.retrieve_only is None and not args.allow_api:
        parser.error("Use --allow-api somente após autorizar o envio e verificar a cobrança do projeto.")

    if operational:
        try:
            from rag.faq_retrieval import FaqRetriever
            from rag.index_faq import open_faq

            retriever = FaqRetriever(open_faq(name=args.collection), max_distance=args.max_distance)
            if args.retrieve_only is not None:
                records = retriever.retrieve(args.retrieve_only)
                if not records:
                    print(NO_CONTEXT)
                for record in records:
                    print(f"[{record.identifier}] {record.answer}")
                return
        except Exception as error:
            parser.exit(2, f"Falha ao abrir ou consultar a FAQ: {error}\n")

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        settings = load_settings(selected=args.provider)
        if not operational:
            retriever = ChromaRetriever(demo_collection())
        session = Session(retriever, make_provider(settings), operational_faq=operational)
    except ConfigurationError as error:
        parser.exit(2, f"{error}\n")
    except Exception:
        if operational:
            parser.exit(2, "Falha ao iniciar o provedor. Verifique as dependências e a configuração.\n")
        parser.exit(2, "Falha ao iniciar. Verifique as dependências e execute python -m examples.seed_demo.\n")

    label = "FAQ operacional Orienta" if operational else "Tutor demonstrativo"
    print(f"{label} ({settings.provider}). Use perguntas sem dados pessoais. Digite /sair para encerrar.")
    while True:
        try:
            question = input("Você: ").strip()
            if question == "/sair":
                return
            if question:
                print(f"Tutor: {session.ask(question)}")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        except ProviderError as error:
            print(error)
        except Exception:
            print("Falha na consulta à base.")


if __name__ == "__main__":
    main()
