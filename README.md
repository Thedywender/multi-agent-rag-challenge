# API RAG com Docker Compose

API para ingestão de documentos e consultas com RAG (Retrieval-Augmented Generation). Executa localmente via Docker Compose, com vector DB (Chroma) em container. Suporta **OpenAI** ou **AWS Bedrock** para embeddings e LLM.

## Desafio técnico

Este repositório inclui um [desafio técnico](CHALLENGE.md) para evolução da aplicação: múltiplas coleções (RH e técnico), arquitetura multi-agente com orquestrador e agentes especialistas. Consulte o `CHALLENGE.md` para os requisitos e exemplos de requisições.

## Arquitetura

```
Cliente → API (FastAPI) → Chroma (vector DB) + OpenAI ou Bedrock (embeddings + LLM)
```

- **POST /documents**: Recebe documento, divide em chunks, gera embeddings e armazena no Chroma
- **POST /ask**: Busca contexto no Chroma, monta prompt e gera resposta via LLM

## Pré-requisitos

- Docker e Docker Compose
- **OpenAI**: Chave da API (`OPENAI_API_KEY`)
- **Bedrock**: Credenciais AWS e acesso ao Bedrock na região configurada

## Como executar

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
docker compose up -d
```

4. Aguarde alguns segundos para o Chroma inicializar. A API estará disponível em `http://localhost:8000`.

## Endpoints

### POST /documents

Envia um documento para indexação.

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "O prazo de pagamento é de 30 dias. O cliente deve efetuar o pagamento até a data do vencimento."}'
```

Resposta:

```json
{
  "doc_id": "uuid-do-documento",
  "chunks_count": 1
}
```

### POST /ask

Faz uma pergunta com base nos documentos indexados.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual é o prazo de pagamento?"}'
```

Resposta:

```json
{
  "answer": "O prazo de pagamento é de 30 dias.",
  "sources": [
    {
      "document": "O prazo de pagamento é de 30 dias...",
      "metadata": {"doc_id": "..."}
    }
  ]
}
```

### GET /health

Health check da API.

```bash
curl http://localhost:8000/health
```

## Script de migração

O script `scripts/migration.py` insere 5 documentos de exemplo em cada domínio (RH e técnico) via `POST /documents`. Útil para popular o banco após implementar as múltiplas coleções descritas no [CHALLENGE.md](CHALLENGE.md).

**Pré-requisito:** API rodando e endpoint `/documents` aceitando o campo `domain` (formato do desafio).

```bash
# Com a API em localhost:8000
python scripts/migration.py

# Com URL customizada
python scripts/migration.py --base-url http://localhost:8000
```

A massa de dados inclui documentos sobre: política de férias, benefícios, onboarding, regulamento interno, home office (RH); API de pagamentos, guia de integração, arquitetura, endpoints, autenticação (técnico).

## Variáveis de ambiente

| Variável       | Descrição                    | Obrigatório quando |
| -------------- | ---------------------------- | ------------------ |
| LLM_PROVIDER   | `openai` ou `bedrock`        | sempre (padrão: openai) |
| OPENAI_API_KEY | Chave da API OpenAI          | LLM_PROVIDER=openai |
| AWS_REGION     | Região AWS (ex: us-east-1)   | LLM_PROVIDER=bedrock |
| AWS_ACCESS_KEY_ID | Credencial AWS             | LLM_PROVIDER=bedrock |
| AWS_SECRET_ACCESS_KEY | Credencial AWS          | LLM_PROVIDER=bedrock |
| CHROMA_HOST   | Host do Chroma (Docker)      | - (padrão: chroma) |
| CHROMA_PORT   | Porta do Chroma              | - (padrão: 8000)   |

## Como parar

```bash
docker compose down
```

Para remover também os dados do Chroma:

```bash
docker compose down -v
```

## Estrutura do projeto

```
rag/
├── CHALLENGE.md          # Desafio técnico (multi-agentes, múltiplas coleções)
├── docker-compose.yml    # api + chroma
├── Dockerfile
├── requirements.txt
├── .env.example
├── api.http              # Exemplos de requisições (REST Client)
├── src/
│   ├── main.py           # FastAPI app
│   ├── ingest/handler.py # Lógica de ingestão
│   ├── query/handler.py  # Lógica de consulta (RAG)
│   └── shared/
│       ├── chunking.py   # Divisão em chunks
│       ├── embeddings.py # Embeddings (OpenAI ou Bedrock)
│       ├── chroma_client.py # Cliente Chroma
│       └── llm.py        # LLM (OpenAI ou Bedrock)
├── scripts/
│   └── migration.py      # Insere documentos de exemplo (5 RH + 5 técnico)
└── README.md
```
