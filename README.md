# Orienta RAG · protótipo acadêmico

Protótipo de terminal independente dos repositórios Java e Expo. O modo demonstrativo contém três trechos fictícios. O modo Orienta usa uma FAQ operacional separada. O código de indexação e a coleção original do professor seguem sem identificação.

## Execução

Use Python 3.12 e um ambiente virtual, a partir deste diretório.

No workspace, a pasta do projeto é `projeto-ads-p4-rag`. Para o teste de JWT
Java → Python executado a partir de `projeto-ads-p4-api`, informe o caminho
ao backend por `ORIENTA_RAG_TEST_PROJECT` e selecione um Python com PyJWT por
`ORIENTA_RAG_TEST_PYTHON`.

```sh
export ORIENTA_RAG_TEST_PROJECT="$(cd ../projeto-ads-p4-rag && pwd)"
```

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-http.txt
python -m examples.seed_demo
python -m unittest discover -s tests -v
```

A carga usa upsert com IDs fixos em `data/chroma`. A suíte completa inclui a API e requer `requirements-http.txt`. Para usar apenas a CLI, as dependências de `requirements.txt` são suficientes. Os testes usam diretório temporário para o Chroma e simulam os provedores.

Para uma sessão online autorizada, copie `.env.example` para `.env` e preencha somente a chave necessária. Confira a elegibilidade e as condições de cobrança no projeto do provedor antes de executar com `--allow-api`. Use apenas perguntas fictícias.

```sh
python -m rag.cli --allow-api
python -m rag.cli --provider openai --allow-api
```

Gemini é o padrão. `--provider` tem precedência sobre `RAG_PROVIDER`. `GEMINI_MODEL` tem padrão `gemini-3.1-flash-lite` e `OPENAI_MODEL` tem padrão `gpt-4o-mini`. A tabela do Google descreve a modalidade gratuita para Flash-Lite. A elegibilidade depende do projeto e da região. [Preços da API Gemini](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-lite)

Pergunte “Reserva do Observatório Aurora” e depois “E quando?”. Para exercitar a resposta local sem evidência, pergunte “Xilofone submarino”.

## Fluxo

1. O terminal abre uma sessão em memória, com uma única mensagem `system`.
2. `ChromaRetriever` consulta `faq_colecao` com `query_texts=[pergunta]` e `n_results=2`. A demonstração descarta candidatos com distância cosseno acima de 0,72. Para um follow-up iniciado por “E” com busca vazia, tenta também a última consulta que encontrou evidência junto da pergunta atual.
3. `Session` entrega o histórico e a pergunta atual enriquecida com os trechos ao provedor escolhido. Se a busca ficar sem contexto, devolve uma mensagem local e registra o turno. Se o provedor falhar, o turno fica disponível para reenvio sem acúmulo de mensagens.
4. O histórico guarda pares com pergunta original e resposta, em ordem. Os trechos recuperados são adicionados apenas à chamada corrente. `/sair` encerra a sessão e descarta a memória.

O adaptador OpenAI usa `chat.completions.create()`, com `temperature=0` e papéis `system`, `user` e `assistant`. O Gemini recebe a instrução de sistema em `system_instruction` e mapeia `assistant` para `model`. As escolhas de provedor não afetam a recuperação ou a memória. [OpenAI Chat Completions](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create) · [Geração de texto Gemini](https://ai.google.dev/gemini-api/docs/text-generation) · [Consultas Chroma](https://docs.trychroma.com/docs/querying-collections/query-and-get)

## Diferenças em relação ao exemplo da aula

- O núcleo preserva o cliente persistente, a coleção, a recuperação antes da geração e o formato de histórico da aula.
- A demonstração usa hashing lexical local, o mesmo na indexação e na consulta. Isso permite executar a recuperação sem baixar pesos. O método depende de sobreposição de termos e tem cobertura limitada de sinônimos. O limiar de distância serve ao corpus fictício e ainda precisaria ser avaliado em um corpus maior.
- `sentence-transformers` e `pypdf` ficam para a avaliação do material original. Este corpus usa texto escrito diretamente e um embedding demonstrativo. O modelo, os PDFs e o pipeline da aula seguem pendentes de confirmação.
- O protótipo acrescenta Gemini, um filtro de relevância, tratamento de falhas e testes. O histórico cresce durante a sessão. A janela de contexto e limites de consumo devem ser definidos antes de uso prolongado ou integração.

## FAQ operacional do Orienta

O chatbot tem um modo de conhecimento operacional sobre o aplicativo, com instruções de conta, perfis, busca, mentorias, chat, notificações, assistente e suporte. A geração de conteúdo pedagógico e a ingestão de materiais de estudo ficam para uma missão futura. O modo existente de demonstração continua com sua coleção lexical em `data/chroma`.

### Corpus e arquitetura

[`knowledge/orienta_faq.json`](knowledge/orienta_faq.json) contém 33 registros, validados por inspeção de código. A raiz tem `versao` e `registros`. Cada entrada preserva ID, categoria, pergunta, resposta, evidência, status, escopo de verificação e fontes. Cada fonte registra repositório, caminho, símbolo e commit. Os prazos de agenda seguem o servidor. O corpus exclui dados reais de usuários e funcionalidades pendentes, como cobrança, recuperação de senha esquecida e videoconferência.

`rag/index_faq.py` valida o formato e a unicidade dos IDs e grava pergunta e resposta em um documento por registro. A base operacional fica em `data/chroma_orienta`. O nome da coleção é derivado da versão e do hash do manifesto. O manifesto inclui o hash do corpus, a métrica, o perfil de embeddings e o número de registros. Lotes são gravados por upsert com IDs estáveis. A coleção só fica disponível após a indexação terminar. Uma alteração no JSON gera outra coleção. Guarde junto dos dados o nome da versão anterior para reabri-la via `--collection`. A operação padrão seleciona a coleção correspondente ao JSON atual. A abertura verifica manifesto, configuração do encoder, dimensão e contagem.

A recuperação consulta até dois candidatos no Chroma com `query_texts=[pergunta]` e filtra pela distância cosseno configurada. O provedor recebe somente os candidatos atuais e o histórico da sessão. Ele seleciona um ID ou indica evidência insuficiente. O terminal mostra a resposta canônica e seu ID de fonte. Texto livre e IDs fora dos candidatos são descartados. Sem candidatos, a resposta sai localmente. Esse desenho restringe o texto exibido ao corpus, mas a seleção de um ID incorreto ainda é possível.

### Modelo e preparo local

O adaptador usa `intfloat/multilingual-e5-small` na revisão [`7cf867cae19b6df205046e9c11907797649fb552`](https://huggingface.co/intfloat/multilingual-e5-small/commit/7cf867cae19b6df205046e9c11907797649fb552). O modelo produz 384 dimensões, com normalização L2 e distância cosseno. A mesma configuração atende ao índice e à consulta. Os prefixos são `passage: ` para documentos e `query: ` para perguntas. O carregamento é local, em CPU, com Safetensors e lotes de quatro textos. [Ficha do modelo](https://huggingface.co/intfloat/multilingual-e5-small)

O E5 foi escolhido por seu treinamento voltado à recuperação multilíngue e pelos prefixos específicos de consulta e passagem. Os pesos Safetensors ocupam cerca de 471 MB, além do tokenizer e das dependências. A máquina auditada tem Celeron N4500 com dois núcleos, 3,6 GiB de RAM total e cerca de 694 MiB disponíveis no momento da medição, com 84 GiB de armazenamento livre. O espaço em disco é suficiente para o artefato, enquanto a memória disponível e o uso de swap são restrições para a execução. Meça tempo e pico de RAM após obter os pesos. [Arquivo de pesos](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/model.safetensors)

Após autorização para transferir as dependências e os pesos, instale `requirements-semantic.txt` em um ambiente virtual Linux x86-64 e disponibilize **essa revisão** no cache Hugging Face. O carregamento usa `local_files_only=True`. A máquina desta missão estava sem o checkpoint e sem o runtime semântico. As versões propostas para esse runtime ainda precisam de validação local.

### Indexação e consulta

Quando o cache e o ambiente estiverem prontos, execute estes comandos a partir do diretório do protótipo.

```sh
python -m rag.index_faq
python -m rag.cli --corpus orienta --max-distance 2 --retrieve-only "Consigo trocar o horário da mentoria?"
```

A consulta local acima usa distância máxima 2 apenas para inspecionar os dois candidatos durante a avaliação. Calibre o limiar com perguntas válidas, ambíguas e fora de escopo antes de usá-lo no modo interativo. A CLI exige informar o limiar explicitamente.

Para uso online autorizado, configure `GEMINI_API_KEY` ou `OPENAI_API_KEY` conforme a seção inicial. Para Gemini, mantenha `RAG_PROVIDER=gemini` ou informe `--provider gemini`. Para OpenAI, use `--provider openai`. No comando interativo, substitua `--retrieve-only PERGUNTA` por `--allow-api` e use o limiar calibrado. `/sair` descarta a memória. A opção `--retrieve-only` funciona sem chave ou provedor.

### Testes e diferenças em relação à aula

```sh
python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -p 'model_faq.py' -v
```

O primeiro comando preserva os 14 testes da demonstração e executa testes da FAQ com Chroma real em armazenamento temporário, encoder simulado e APIs simuladas. O segundo exercita o ranking com o modelo real e exige o checkpoint já disponível, sem chamadas a Gemini ou OpenAI. A suíte simulada valida a integração e os casos previstos, sem medir qualidade semântica do E5.

A adaptação mantém Chroma persistente, dois resultados iniciais, recuperação antes da geração e os papéis da conversa. O modo Orienta acrescenta FAQ JSON com proveniência, encoder local em português, manifesto e seleção de uma resposta canônica. O exemplo de FAQ em PDF da aula serve como referência de estrutura. `pypdf` não é necessário para o corpus JSON.

## API HTTP interna

`rag.http_api:create_app` monta a API FastAPI. `rag/http_auth.py` valida a credencial interna, `rag/http_sessions.py` controla propriedade, expiração e turnos, e `rag/http_modes.py` compõe a sessão do modo escolhido. A CLI continua disponível separadamente.

### Ambiente local

Use Python 3.12 e um ambiente virtual. A API local usa loopback, um worker e documentação web desativada. Para uso futuro entre serviços, mantenha o Python em rede privada, sem publicar sua porta, e use TLS entre hosts. O Spring continua responsável pelo acesso público e pela autenticação do aplicativo.

```sh
python -m pip install -r requirements-http.txt
export RAG_INTERNAL_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export RAG_HTTP_MODE=simulated
python -m uvicorn rag.http_api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

| Variável | Valor inicial |
|---|---|
| `RAG_INTERNAL_JWT_SECRET` | Obrigatória, pelo menos 32 bytes. Use um segredo gerado exclusivamente para esta API. |
| `RAG_INTERNAL_JWT_ISSUER` / `RAG_INTERNAL_JWT_AUDIENCE` | `orienta-spring` / `orienta-rag` |
| `RAG_HTTP_MODE` | `simulated` |
| `RAG_HTTP_IDLE_SECONDS` | `900` segundos de inatividade |
| `RAG_HTTP_MAX_SESSIONS` / `RAG_HTTP_MAX_TURNS` | `100` conversas no processo / `20` turnos na janela de histórico |
| `RAG_HTTP_MAX_MESSAGES` / `RAG_HTTP_MAX_CONCURRENT` | `128` IDs reservados por conversa / `4` turnos simultâneos no processo |

### Autenticação e contrato

Cada rota, inclusive saúde, exige `Authorization: Bearer <credencial-interna>`. A credencial usa HS256 com segredo dedicado e claims `iss`, `aud`, `sub`, `iat`, `exp` e `scope`. A duração máxima é de 120 segundos. `iat` e `exp` são segundos Unix inteiros. O escopo exigido é `rag:internal`. Na integração futura, o Spring deve derivar um `sub` pseudônimo estável do principal autenticado e assinar uma credencial interna. O cliente do aplicativo envia seu JWT somente ao Spring.

| Rota sob `/internal/v1` | Resposta |
|---|---|
| `POST /conversations` | Recebe `{}`. Retorna `201` com `conversationId`, `mode` e `expiresAt`. |
| `POST /conversations/{conversationId}/messages` | Recebe `messageId` UUID e `question` com até 1.000 caracteres após trim. Retorna `200` com `conversationId`, `messageId`, `mode`, `outcome`, `answer` e `evidence`. |
| `DELETE /conversations/{conversationId}` | Encerra e descarta a conversa. Retorna `204`. |
| `GET /health/live` | Retorna `200` se o processo responde. |
| `GET /health/ready` | Retorna `200` com o modo pronto ou `503` com o motivo da indisponibilidade. |

A resposta `answered` inclui o texto canônico e uma evidência com `id`, `corpus` e `corpusVersion`. `no_evidence` devolve uma orientação local com lista vazia. O contrato de erro é `{"error":{"code":"...","message":"...","retryable":false}}`. Credencial inválida retorna `401`, escopo insuficiente `403`, conversa inexistente, expirada ou de outro sujeito `404`, conflito ou turno em andamento `409`, entrada inválida `422`, limites `429`, recurso indisponível `503` e timeout do provedor `504`.

Exemplo com duas janelas de terminal. Na janela do cliente, disponibilize o mesmo segredo de desenvolvimento e execute os comandos a partir do projeto. Use outro token quando os 60 segundos expirarem.

```sh
token="$(python - <<'PY'
import os
import time
import jwt
now = int(time.time())
print(jwt.encode({
    "iss": os.environ.get("RAG_INTERNAL_JWT_ISSUER", "orienta-spring"),
    "aud": os.environ.get("RAG_INTERNAL_JWT_AUDIENCE", "orienta-rag"),
    "sub": "usuario-teste",
    "iat": now,
    "exp": now + 60,
    "scope": "rag:internal",
}, os.environ["RAG_INTERNAL_JWT_SECRET"], algorithm="HS256"))
PY
)"
base=http://127.0.0.1:8000/internal/v1
conversation="$(curl -fsS "$base/conversations" -H "Authorization: Bearer $token" \
  -H 'Content-Type: application/json' -d '{}' | python -c 'import json,sys; print(json.load(sys.stdin)["conversationId"])')"
message="$(python -c 'import uuid; print(uuid.uuid4())')"
curl -fsS "$base/conversations/$conversation/messages" -H "Authorization: Bearer $token" \
  -H 'Content-Type: application/json' \
  -d "{\"messageId\":\"$message\",\"question\":\"Consigo trocar o horário da mentoria?\"}"
curl -fsS -X DELETE "$base/conversations/$conversation" -H "Authorization: Bearer $token"
```

A memória é volátil. O servidor gera a conversa e a vincula ao sujeito autenticado. Cada conversa processa um turno por vez. Repetir um ID com a mesma pergunta retorna a resposta armazenada. Usar o mesmo ID com outra pergunta retorna conflito. Após falha do provedor, a tentativa pode ser repetida com o mesmo ID e pergunta sem duplicar o histórico. A janela preserva a instrução de sistema e os pares mais recentes. Reiniciar o processo encerra as conversas.

### Modos e testes

- `demonstrative` usa uma cópia temporária do Chroma fictício existente e seleciona a primeira evidência recuperada. Se ainda não houver um índice demonstrativo em um ambiente novo, use a rotina de carga da seção inicial.
- `simulated` usa a FAQ Orienta e um recuperador de perguntas normalizadas e exemplos fixos, sem provedor externo. Execute-o em desenvolvimento ou homologação. **Esse modo não comprova a qualidade da recuperação semântica.**
- `semantic_real` está preparado como modo indisponível nesta entrega. Se existir um índice, valida uma cópia contra o manifesto. Um manifesto incompatível produz `INDEX_INCOMPATIBLE`. Sem essa falha, retorna `MODEL_UNAVAILABLE`. A inicialização e as rotas de saúde não carregam pesos ou chamam provedores.

```sh
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
```

A suíte usa FastAPI TestClient, bases demonstrativas temporárias e provedores locais ou simulados. A suíte opcional de pesos reais fica separada. A integração com o Spring deve definir a rota autenticada, a emissão da credencial interna, a contingência determinística e os prazos entre serviços. O armazenamento em memória pressupõe um único worker. Múltiplas réplicas exigem armazenamento compartilhado e coordenação de turnos.
