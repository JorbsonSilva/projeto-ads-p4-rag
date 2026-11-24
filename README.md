# Tutor Orienta · RAG acadêmico

Projeto da disciplina de Machine Learning que aplica recuperação aumentada por geração (RAG) a dúvidas operacionais do Orienta. O Tutor explica como usar o aplicativo e fundamenta a resposta em registros da FAQ. Triagem, recomendações e agendamentos continuam no assistente determinístico e nas telas próprias do Orienta.

## Arquitetura

| Componente | Branch | Papel |
|---|---|---|
| [Expo Web](https://github.com/JorbsonSilva/projeto-ads-p4-mobile/tree/feature/orienta-rag) | `feature/orienta-rag` | Login do aluno, interface do Tutor, origem da resposta e evidências |
| [Spring Boot](https://github.com/JorbsonSilva/projeto-ads-p4-api/tree/feature/orienta-rag) | `feature/orienta-rag` | API pública autenticada e emissão de JWT interno de curta duração |
| [Python RAG](https://github.com/JorbsonSilva/projeto-ads-p4-rag/tree/main) | `main` | FastAPI privada, memória por sessão, embeddings, ChromaDB persistente e geração |

O Expo envia o token do aplicativo somente ao Spring. O Spring identifica o usuário e assina uma credencial interna para o FastAPI. O FastAPI consulta o ChromaDB e envia ao Gemini apenas o histórico e os candidatos do turno. O resultado volta ao Expo com o texto e os IDs das evidências, passando pelo Spring.

## Preparação

Clone os três projetos lado a lado e selecione as branches da tabela. No Python RAG, use Python 3.12 e um ambiente virtual.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
```

Para o piloto, siga a instalação ONNX abaixo. O modo simulado pode usar `requirements-http.txt`. Para reproduzir a coleção fictícia em outro ambiente, execute `python -m examples.seed_demo`. A coleção operacional tem um diretório próprio.

## FAQ operacional do Orienta

O chatbot tem um modo de conhecimento operacional sobre o aplicativo, com instruções de conta, perfis, busca, mentorias, chat, notificações, assistente e suporte. A geração de conteúdo pedagógico e a ingestão de materiais de estudo ficam para uma missão futura. O modo existente de demonstração continua com sua coleção lexical em `data/chroma`.

### Corpus e arquitetura

[`knowledge/orienta_faq.json`](knowledge/orienta_faq.json) contém 33 registros, validados por inspeção de código. A raiz tem `versao` e `registros`. Cada entrada preserva ID, categoria, pergunta, resposta, evidência, status, escopo de verificação e fontes. Cada fonte registra repositório, caminho, símbolo e commit. Os prazos de agenda seguem o servidor. O corpus exclui dados reais de usuários e funcionalidades pendentes, como cobrança, recuperação de senha esquecida e videoconferência.

`rag/index_faq.py` valida o formato e a unicidade dos IDs e grava pergunta e resposta em um documento por registro. A base operacional fica em `data/chroma_orienta`. O nome da coleção é derivado da versão e do hash do manifesto. O manifesto inclui o hash do corpus, a métrica, o perfil de embeddings e o número de registros. Lotes são gravados por upsert com IDs estáveis. A coleção só fica disponível após a indexação terminar. Uma alteração no JSON gera outra coleção. Guarde junto dos dados o nome da versão anterior para reabri-la via `--collection`. A operação padrão seleciona a coleção correspondente ao JSON atual. A abertura verifica manifesto, configuração do encoder, dimensão e contagem.

A recuperação usa `query_texts` com `n_results=2` e filtra pela distância cosseno. O provedor recebe o histórico `system`, `user` e `assistant` e a pergunta atual com os candidatos. O system prompt define a persona do Tutor, pede português brasileiro e limita a resposta aos trechos recuperados. O provedor devolve um JSON com `id` e `resposta`. O serviço só publica evidências de IDs recuperados. Quando faltam candidatos ou o modelo se abstém, a resposta é local. O prompt limita o uso do histórico à interpretação da pergunta. A checagem do ID valida a proveniência, mas a fidelidade do texto gerado precisa ser avaliada.

### Modelo e execução

O perfil padrão `sentence_transformers` usa `intfloat/multilingual-e5-small` na revisão `7cf867cae19b6df205046e9c11907797649fb552`. Instale `requirements-semantic.txt` e disponibilize essa revisão no cache local para usar esse perfil. O perfil `onnx`, validado no Celeron N4500, usa o mesmo E5 convertido para uint8, em CPU com uma thread. Ambos usam 384 dimensões, prefixos `query: ` e `passage: ` e normalização L2. O perfil ONNX usa média com máscara de atenção e inclui a categoria junto à pergunta e resposta indexadas. Os manifestos mantêm os índices separados. [Modelo original](https://huggingface.co/intfloat/multilingual-e5-small) · [Artefatos ONNX](https://huggingface.co/Xenova/multilingual-e5-small/tree/761b726dd34fb83930e26aab4e9ac3899aa1fa78)

Para o perfil leve, use um ambiente virtual Python 3.12, `requirements-onnx.txt` e só os dois artefatos abaixo (135,1 MB).

```sh
python -m pip install -r requirements-onnx.txt
revision=761b726dd34fb83930e26aab4e9ac3899aa1fa78
base=https://huggingface.co/Xenova/multilingual-e5-small/resolve/$revision
mkdir -p models/e5-onnx
curl -fL "$base/tokenizer.json" -o models/e5-onnx/tokenizer.json
curl -fL "$base/onnx/model_uint8.onnx" -o models/e5-onnx/model_uint8.onnx
export RAG_EMBEDDING_BACKEND=onnx RAG_MAX_DISTANCE=0.14
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m rag.index_faq
python -m rag.cli --corpus orienta --max-distance 0.14 --retrieve-only "Consigo trocar o horário da mentoria?"
```

O limiar 0,14 é inicial para este índice ONNX. A distância sozinha pode aproximar assuntos externos ao Orienta. Pedidos de emissão ou renovação de passaporte são descartados antes da recuperação. Outros casos próximos do limiar ainda dependem da abstenção do provedor. Para inspecionar os dois candidatos sem filtro de distância, use `--max-distance 2`.

Para geração real, copie `.env.example` para `.env` local, ignorado pelo Git. O serviço carrega esse arquivo com `python-dotenv`. Configure `GEMINI_API_KEY` (padrão) ou `OPENAI_API_KEY` no ambiente ou `.env`. Selecione o modelo com `GEMINI_MODEL` ou `OPENAI_MODEL` e substitua `--retrieve-only PERGUNTA` por `--allow-api`. `/sair` descarta a memória. O piloto validou Gemini `gemini-3.1-flash-lite`.

No teste HTTP real, “Como faço para cancelar uma mentoria?” retornou o prazo de duas horas e a exigência de informar o motivo, com evidência `M08`. “E qual é o prazo?” manteve `M08` na mesma sessão e retornou ausência de evidência em uma sessão nova. A remarcação retornou `M07`, com antecedência de 24 horas para o novo horário e verificação de conflitos. Clima e emissão de passaporte retornaram ausência de evidência.

### Testes e diferenças em relação à aula

```sh
python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -p 'model_faq.py' -v
```

Na entrega, os **34 testes Python** da primeira linha passaram. Eles incluem Chroma real em armazenamento temporário, encoders de teste e provedores simulados. A segunda linha é opcional e usa o checkpoint local do perfil Sentence Transformers. A validação com ONNX e Gemini reais foi realizada separadamente, pela API e pelo Expo Web.

A adaptação mantém Chroma persistente, dois resultados iniciais, recuperação antes da geração e os papéis da conversa. O modo Orienta acrescenta FAQ JSON com proveniência, encoder local em português, manifesto e validação das evidências retornadas. O exemplo de FAQ em PDF da aula serve como referência de estrutura. `pypdf` não é necessário para o corpus JSON.

## API HTTP interna

`rag.http_api:create_app` monta a API FastAPI. `rag/http_auth.py` valida a credencial interna, `rag/http_sessions.py` controla propriedade, expiração e turnos, e `rag/http_modes.py` compõe a sessão do modo escolhido. A CLI continua disponível separadamente.

### Ambiente local

Use Python 3.12 e um ambiente virtual. A API local usa loopback, um worker e documentação web desativada. Para uso futuro entre serviços, mantenha o Python em rede privada, sem publicar sua porta, e use TLS entre hosts. O Spring continua responsável pelo acesso público e pela autenticação do aplicativo.

```sh
python -m pip install -r requirements-http.txt
export RAG_INTERNAL_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export RAG_HTTP_MODE=semantic_real RAG_EMBEDDING_BACKEND=onnx RAG_MAX_DISTANCE=0.14
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

A resposta `answered` inclui o texto da resposta e uma evidência com `id`, `corpus` e `corpusVersion`. `no_evidence` devolve uma orientação local com lista vazia. O contrato de erro é `{"error":{"code":"...","message":"...","retryable":false}}`. Credencial inválida retorna `401`, escopo insuficiente `403`, conversa inexistente, expirada ou de outro sujeito `404`, conflito ou turno em andamento `409`, entrada inválida `422`, limites `429`, recurso indisponível `503` e timeout do provedor `504`.

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
- `semantic_real` abre o índice operacional persistente e usa o perfil definido por `RAG_EMBEDDING_BACKEND`, o limiar obrigatório `RAG_MAX_DISTANCE` e um provedor configurado. A inicialização carrega o encoder antes de indicar prontidão. Configure o segredo JWT interno no ambiente e execute o comando Uvicorn acima com `RAG_HTTP_MODE=semantic_real`. O modo devolve `503` se índice, modelo ou provedor estiverem indisponíveis e não muda para o modo simulado. O Spring mapeia `semantic_real` para `RAG`. Para o piloto, use `ORIENTA_RAG_RESPONSE_TIMEOUT=10s` no Spring e mantenha o prazo local de 12 s do Expo. A geração real ultrapassou o padrão de 3 s do Spring; os prazos do piloto devem ser reavaliados se a latência do provedor variar.

```sh
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
```

A suíte usa FastAPI TestClient, bases demonstrativas temporárias e provedores locais ou simulados. A suíte opcional de pesos reais fica separada. O armazenamento em memória pressupõe um único worker. Múltiplas réplicas exigem armazenamento compartilhado e coordenação de turnos.

### Piloto integrado no Expo Web

Use JDK 17, Node.js compatível com Expo SDK 54 (Node 22 no piloto) e um PostgreSQL de teste isolado com dados sintéticos. Execute `npm ci` no frontend e configure `DB_URL`, `DB_USER` e `DB_PASSWORD` no processo Spring. No Python, exporte `RAG_INTERNAL_JWT_SECRET` e inicie o Uvicorn em `semantic_real` conforme acima. No Spring, use `ORIENTA_RAG_ENABLED=true`, `ORIENTA_RAG_BASE_URL=http://127.0.0.1:8000`, `ORIENTA_RAG_INTERNAL_SECRET` igual ao segredo do Python e um `ORIENTA_RAG_SUBJECT_SECRET` distinto. Abra o Expo Web em `http://localhost:8081`, origem aceita pelo `CorsConfig` atual do Spring. A origem `http://127.0.0.1:8081` é distinta para CORS.

```sh
# No backend, após configurar o banco de teste e os segredos no processo.
ORIENTA_RAG_ENABLED=true ORIENTA_RAG_BASE_URL=http://127.0.0.1:8000 \
  ORIENTA_RAG_RESPONSE_TIMEOUT=10s ./mvnw spring-boot:run -Dspring-boot.run.profiles=dev

# No frontend, sem herdar segredos do Spring ou Python.
EXPO_PUBLIC_TUTOR_ENABLED=true EXPO_PUBLIC_API_URL=http://localhost:8080 \
  ./node_modules/.bin/expo start --web --offline --max-workers 1
```

Entre com uma conta sintética de aluno do seed `dev`, descrito no README do backend, e abra **Dúvidas sobre o Orienta**. O Tutor mostra respostas reais como **Base de conhecimento**, origem `RAG` no contrato público. O histórico é volátil e fica restrito à conversa. Em máquinas com pouca RAM, prefira exportar o bundle Web antes de iniciar os serviços e confirme a memória disponível em cada etapa.


Na demonstração completa no Chrome, o login de aluno, a abertura pelo atalho, a resposta com evidência, a continuidade e o encerramento passaram. O cancelamento levou 8,61 s e a continuação, 4,06 s no aplicativo. Pedir emissão de passaporte retornou ausência de evidência em 0,36 s. A triagem determinística também respondeu com HTTP 200. Os 10 s do Spring foram uma configuração temporária do piloto.

O corpus cobre apenas a FAQ operacional existente. O filtro de relevância é heurístico e o ID validado comprova a origem do candidato, sem garantir por si só a fidelidade de cada frase. A latência depende da máquina, rede e provedor. A demonstração integrada foi feita no Expo Web. Para a apresentação, veja [DEMONSTRACAO.md](DEMONSTRACAO.md).
