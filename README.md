# API RAG com Multi-Agentes/Docker Compose

API para ingestão de documentos e consultas com RAG (Retrieval-Augmented Generation). Executa localmente via Docker Compose, com vector DB (Chroma) em container. Suporta **OpenAI** ou **AWS Bedrock** para embeddings e LLM.

## Arquitetura

```
Cliente → API (FastAPI) → Orquestrador → Agente RH | Agente Técnico
                                ↓
                        Chroma (rh_docs/tecnico_docs) + OpenAI ou Bedrock
```

- **POST /documents**: Recebe documento e `domain` (`rh` ou `tecnico`), divide em chunks, gera embeddings e armazena no Chroma, armazena cada `domain` separado, para busca ser idempendente.

- **POST /ask**: Orquestra a pergunta (`rh`, `tecnico` ou `geral`), consulta as coleções corretas e gera resposta via LLM

## Decisões de design

- Coleções separadas no Chroma por domínio (`rh_docs` e `tecnico_docs`) com metadados `doc_id` e `domain`.
- Orquestração com LangGraph (`embed -> route -> answer`) e registry de agentes especialistas.
- Classificação híbrida: palavras-chave para casos diretos e fallback LLM em perguntas ambíguas.
- Compatibilidade de provedor via `LLM_PROVIDER` para embeddings e geração (`openai` ou `bedrock`).

## Fluxo de roteamento

1. A API recebe a pergunta em `POST /ask`.
2. O orquestrador classifica em `rh`, `tecnico` ou `geral`.
3. Para `rh` e `tecnico`, encaminha ao agente especialista do domínio.
4. Para `geral`, aplica fallback e consulta as duas coleções (`rh_docs` e `tecnico_docs`), retornando as fontes combinadas.

## Documentação técnica detalhada

- [Arquitetura detalhada](docs/ARCHITECTURE.md)
- [Decisões técnicas e fluxo de roteamento](docs/TECHNICAL_DESIGN.md)

## Pré-requisitos

- Docker e Docker Compose
- **OpenAI**: Chave da API (`OPENAI_API_KEY`)
- **Bedrock**: Credenciais AWS e acesso ao Bedrock na região configurada

## Como executar com Docker Compose

1. Copie o arquivo de exemplo de variáveis de ambiente:

```bash
cp .env.example .env
```

2. Edite o `.env` e configure o provedor escolhido:

**Opção A - OpenAI (padrão):**

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-sua-chave-aqui
```

**Opção B - AWS Bedrock:**

```
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=sua-access-key
AWS_SECRET_ACCESS_KEY=sua-secret-key
```

**Importante:** O arquivo `.env` é obrigatório. Para OpenAI, a chave não pode ser placeholder (`sk-your-...`). Para Bedrock, é necessário ter acesso à AWS configurado (credenciais ou IAM role). Ao trocar de provedor, execute `docker compose down -v` para limpar os dados do Chroma, pois os embeddings têm dimensões diferentes.

3. Suba os serviços:

```bash
# Primeiro start (ou após mudança de dependências)
docker compose up -d --build

# Starts seguintes (sem rebuild)
docker compose up -d
```

4. Aguarde alguns segundos para o Chroma inicializar. A API estará disponível em `http://localhost:8000`.

5. (Opcional) Comandos úteis no dia a dia:

```bash
# Pausar containers sem remover
docker compose stop

# Retomar containers parados
docker compose start

# Derrubar e remover volumes (zera dados do Chroma)
docker compose down -v
```

## Endpoints

### POST /documents

Envia um documento para indexação na coleção do domínio informado.

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "A política de férias permite divisão em até 3 períodos, com aprovação do RH.", "domain": "rh"}'
```

Resposta:

```json
{
  "doc_id": "uuid-do-documento",
  "chunks_count": 1,
  "domain": "rh",
  "already_exists": false
}
```

### POST /ask

Faz uma pergunta; o orquestrador classifica e rota para o agente especialista.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual é a política de férias?"}'
```

Resposta:

```json
{
  "answer": "A política de férias permite divisão em até 3 períodos com aprovação do RH.",
  "sources": [
    {
      "document": "A política de férias permite divisão em até 3 períodos...",
      "metadata": { "doc_id": "...", "domain": "rh" }
    }
  ],
  "routed_domain": "rh"
}
```

### GET /health

Health check da API.

```bash
curl http://localhost:8000/health
```

## Popular o Chroma com `scripts/migration.py`

O script `scripts/migration.py` popula as coleções com dados de exemplo via `POST /documents`:

- 5 documentos no domínio `rh`
- 5 documentos no domínio `tecnico`

Isso prepara rapidamente o ambiente para testar o roteamento multi-agente sem inserir documentos manualmente.

### Pré-requisitos

- API em execução (`docker compose up -d`)
- Endpoint `/documents` aceitando o campo `domain` (já implementado neste projeto)

### Executar migração

```bash
# URL padrão (http://localhost:8000)
python3 scripts/migration.py

# URL customizada
python3 scripts/migration.py --base-url http://localhost:8000
```

Saída esperada (resumo):

```text
Conectando em http://localhost:8000
Inserindo 10 documentos (5 RH + 5 técnico)...
...
--- Resumo ---
Sucesso: 10/10
Migração concluída com sucesso.
```

Se quiser repopular do zero:

```bash
docker compose down -v
docker compose up -d --build
python3 scripts/migration.py
```

A massa de dados inclui temas de RH (férias, benefícios, onboarding, regulamento interno, home office) e temas técnicos (API de pagamentos, integração, arquitetura, endpoints e autenticação).

## Como fazer queries após popular o Chroma

Depois da migração, use `POST /ask` normalmente. O orquestrador classifica a pergunta e retorna o domínio roteado em `routed_domain`.

### Query de RH

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual é a política de férias?"}'
```

Esperado: `routed_domain = "rh"`.

### Query técnica

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Como autenticar na API de pagamentos?"}'
```

Esperado: `routed_domain = "tecnico"`.

### Query ambígua (fallback geral)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais regras de benefícios e endpoint de pagamentos?"}'
```

Esperado: `routed_domain = "geral"` com fontes combinadas dos dois domínios quando aplicável.

## Testes

Cobertura de fluxos críticos:

- Ingestão com deduplicação e controle de corrida
- Orquestração e roteamento (`rh`, `tecnico`, `geral`)
- Agentes especialistas por domínio
- Contratos de request/response e handlers de erro
- Compatibilidade de provider (`openai`/`bedrock`)

Execute:

```bash
pytest -q
```

## Variáveis de ambiente

| Variável              | Descrição                  | Obrigatório quando      |
| --------------------- | -------------------------- | ----------------------- |
| LLM_PROVIDER          | `openai` ou `bedrock`      | sempre (padrão: openai) |
| OPENAI_API_KEY        | Chave da API OpenAI        | LLM_PROVIDER=openai     |
| AWS_REGION            | Região AWS (ex: us-east-1) | LLM_PROVIDER=bedrock    |
| AWS_ACCESS_KEY_ID     | Credencial AWS             | LLM_PROVIDER=bedrock    |
| AWS_SECRET_ACCESS_KEY | Credencial AWS             | LLM_PROVIDER=bedrock    |
| CHROMA_HOST           | Host do Chroma (Docker)    | - (padrão: chroma)      |
| CHROMA_PORT           | Porta do Chroma            | - (padrão: 8000)        |

## Estrutura do projeto

```

multi-agent-rag-challenge/
├── CHALLENGE.md # Desafio técnico (multi-agentes, múltiplas coleções)
├── docker-compose.yml # api + chroma
├── Dockerfile
├── requirements.txt
├── .env.example
├── api.http # Exemplos de requisições (REST Client)
├── docs/
│ ├── architecture.drawio # Diagrama editável da arquitetura
│ ├── ARCHITECTURE.md # Fluxos e componentes da solução
│ └── TECHNICAL_DESIGN.md # Decisões de design e roteamento
├── src/
│ ├── main.py # FastAPI app
│ ├── ingest/handler.py # Lógica de ingestão
│ ├── query/handler.py # Lógica de consulta (RAG)
│ ├── orchestrator/ # Classificação e roteamento
│ ├── agents/ # Agentes especialistas RH e técnico
│ └── shared/
│ ├── chunking.py # Divisão em chunks
│ ├── embeddings.py # Embeddings (OpenAI ou Bedrock)
│ ├── chroma_client.py # Cliente Chroma
│ └── llm.py # LLM (OpenAI ou Bedrock)
├── scripts/
│ └── migration.py # Insere documentos de exemplo (5 RH + 5 técnico)
├── tests/ # Testes unitários (ingestão, roteamento, agentes e contrato)
└── README.md

```
