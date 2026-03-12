# Documentação Técnica

## Objetivo

Evoluir uma API RAG monolítica para um modelo multi-coleção e multi-agente com roteamento por domínio, mantendo compatibilidade com OpenAI e AWS Bedrock.

## Decisões de design

### 1. Separação por domínio no armazenamento

- Coleções lógicas por domínio: `rh` e `tecnico`.
- Mapeamento físico no Chroma: `rh_docs` e `tecnico_docs`.
- Metadados por chunk: `doc_id` (agrupamento), `domain` (rastreabilidade), `content_hash` (deduplicação) e `chunk_index` (ordenação).

Motivação: isolamento de contexto por domínio e redução de ruído sem depender apenas de prompts.

### 2. Ingestão idempotente com proteção de corrida

Pipeline de `handle_ingest`:

1. validação (`content`, `domain`);
2. hash SHA-256 do conteúdo;
3. pré-checagem por `content_hash`;
4. tentativa de inserção;
5. fallback de corrida: se houver duplicidade na inserção, reconsulta por hash e retorna o já persistido.

Trade-off: evita duplicação silenciosa sem lock distribuído explícito; custo é uma reconsulta em conflito.

### 3. Orquestração com LangGraph

Grafo fixo para consulta:

- `embed`: vetoriza pergunta;
- `route`: classifica domínio;
- `answer`: delega ao agente ou aplica fallback geral.

Motivação: fluxo explícito, testável por nó e extensível para novos nós (memória, guardrails, rerank).

### 4. Classificador híbrido (keyword + LLM)

`route_question` combina heurística e LLM:

- decisão imediata quando há sinal forte por palavras-chave;
- fallback para LLM quando há ambiguidade;
- fallback final para `geral` em caso de erro do provedor.

Trade-off: melhor latência média do que classificar tudo com LLM e maior robustez para casos ambíguos.

### 5. Contrato dos agentes especialistas

Cada agente:

- consulta apenas sua coleção (`rh` ou `tecnico`);
- gera resposta com as fontes usadas;
- retorna `routed_domain` consistente.

Motivação: previsibilidade no comportamento e facilidade de auditoria de roteamento.

### 6. Compatibilidade de provedor

`LLM_PROVIDER` define backend único para embeddings e geração:

- `openai`
- `bedrock`

Motivação: facilitar deploy em ambientes diferentes sem alterar fluxo de domínio.

## Fluxo de roteamento detalhado

1. Normalização de texto (lowercase, remoção de acentos, whitespace).
2. Score de palavras-chave RH e técnico.
3. Regras de decisão: score unilateral roteia diretamente; margem mínima entre scores roteia diretamente; empate/ambiguidade usa classificador LLM.
4. Erro no classificador LLM: retorna `geral`.

## Observabilidade e tratamento de erro

- `ValidationInputError` (campo, código, mensagem) para erros de entrada.
- Handler global para ValueError: validação retorna HTTP 400 e erro de processamento retorna HTTP 503.
- Handler genérico retorna HTTP 500 para falhas inesperadas.
- Logs estruturados em pontos críticos de ingestão e Chroma.

## Estratégia de testes

Cobertura principal:

- `tests/test_ingest_handler.py`: deduplicação por hash, inserção nova, corrida com recheck e erro em conflito não resolvido.
- `tests/test_router_and_agent.py`: classificação por keyword, fallback `geral` e isolamento de domínio dos agentes.
- `tests/test_langgraph_router.py`: nós do grafo, delegação para especialista, fallback `geral` e invocação do grafo compilado.
- `tests/test_query_handler.py`: validação de pergunta e repasse correto para o LangGraph.
- `tests/test_provider_compat.py`: contrato de provider (`openai`/`bedrock`) e erro para valor inválido.
- `tests/test_api_contract.py`: contratos de request/response e handlers de erro.

## Extensões futuras

- adicionar mais domínios (ex.: financeiro, jurídico) com novos agentes no registry;
- adicionar re-ranker para melhorar precisão no fallback `geral`;
- adicionar testes E2E com ambiente Docker completo (API + Chroma).
