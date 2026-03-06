# Desafio Técnico — Desenvolvedor Gen AI

## Sobre a vaga

Vaga para **Desenvolvedor de Tecnologias com Gen AI**, com foco em evolução de sistemas RAG e arquitetura multi-agente.

---

## Contexto

Você receberá o código de uma aplicação RAG (Retrieval-Augmented Generation) em Python/FastAPI que hoje:

- Ingerir documentos via `POST /documents`
- Responder perguntas via `POST /ask` usando busca vetorial no Chroma e geração com LLM (OpenAI ou AWS Bedrock)

Sua tarefa é evoluir essa aplicação para um sistema com **múltiplas coleções** e **multi-agentes**, com orquestração e roteamento por domínio.

### Execução da aplicação e migração

Consulte o [README](README.md) para instruções de como executar a aplicação (Docker Compose, variáveis de ambiente, etc.).

O repositório inclui um **script de migração** (`scripts/migration.py`) que insere automaticamente 5 documentos em cada domínio (RH e técnico) via `POST /documents`. Use-o para popular as coleções após implementar as múltiplas coleções. O README contém os detalhes de como executar a migração.

---

## Objetivos do desafio

### 1. Múltiplas coleções

Implementar **duas coleções** no Chroma:

- **`rh`** — documentos de Recursos Humanos (políticas, benefícios, processos internos, etc.)
- **`tecnico`** — documentos técnicos (APIs, arquitetura, código, documentação de produto, etc.)

- O endpoint de ingestão deve permitir informar em qual coleção o documento será indexado.
- Cada coleção deve ter metadados adequados ao seu domínio.

### 2. Arquitetura multi-agente

Implementar:

- **Agente orquestrador**: classifica a pergunta do usuário e decide qual agente especialista deve responder.
- **Agente especialista RH**: responde perguntas sobre Recursos Humanos usando a coleção `rh`.
- **Agente especialista Técnico**: responde perguntas técnicas usando a coleção `tecnico`.

### 3. Fluxo de roteamento

Quando o usuário fizer uma pergunta:

1. O **orquestrador** classifica a intenção (ex.: `rh`, `tecnico` ou `geral`).
2. A pergunta é encaminhada ao **agente especialista** do domínio correspondente.
3. O agente especialista usa RAG na coleção apropriada e retorna a resposta.

Exemplos:

- *"Qual é a política de férias?"* → orquestrador classifica como `rh` → agente RH → busca em `rh`.
- *"Como integrar a API de pagamentos?"* → orquestrador classifica como `tecnico` → agente técnico → busca em `tecnico`.
- Em caso de dúvida, o orquestrador pode consultar ambos os agentes ou seguir uma regra de fallback definida por você.

---

## Requisitos técnicos

### Obrigatórios

- [ ] Ingestão com seleção de coleção (`rh` ou `tecnico`).
- [ ] Orquestrador que classifica a pergunta e roteia para o agente correto.
- [ ] Agente especialista RH que consulta apenas a coleção `rh`.
- [ ] Agente especialista Técnico que consulta apenas a coleção `tecnico`.
- [ ] Manter compatibilidade com OpenAI e AWS Bedrock (embeddings + LLM).
- [ ] API REST funcional (FastAPI).

### Desejáveis

- [ ] Uso de LangChain ou LangGraph para orquestração de agentes.
- [ ] Testes unitários e/ou de integração.
- [ ] Documentação da API (OpenAPI/Swagger) atualizada.
- [ ] README com instruções de execução e exemplos de uso.

---

## Exemplos de requisições

### Ingestão de documentos — `POST /documents`

O endpoint deve receber `content` (texto do documento) e `domain` (coleção de destino: `rh` ou `tecnico`).

**Exemplo — documento de RH:**

```json
{
  "content": "A inteligência artificial é um campo da ciência da computação que busca criar sistemas capazes de realizar tarefas que normalmente requerem inteligência humana, como reconhecimento de voz, tomada de decisões e tradução entre idiomas.",
  "domain": "rh"
}
```

**Exemplo — documento técnico:**

```json
{
  "content": "A API de pagamentos aceita requisições POST no endpoint /v1/payments. O payload deve incluir amount, currency e customer_id. A autenticação é feita via Bearer token no header Authorization.",
  "domain": "tecnico"
}
```

### Consulta — `POST /ask`

O endpoint de consulta permanece com a mesma estrutura. O orquestrador classifica a pergunta e encaminha ao agente especialista correspondente.

```json
{
  "question": "Qual é a política de férias?"
}
```

```json
{
  "question": "Como integrar a API de pagamentos?"
}
```

---

## Entregáveis

1. **Código-fonte**  
   Repositório Git com o código evoluído

2. **README**  
   Instruções para:
   - Configuração (`.env`, variáveis necessárias)
   - Execução local e com Docker
   - Exemplos de chamadas à API (ingestão e consulta)

3. **Documentação breve**  
   Arquivo (ex.: `DESAFIO.md` ou seção no README) descrevendo:
   - Arquitetura adotada (diagrama ou texto)
   - Decisões de design
   - Como o orquestrador classifica e roteia as perguntas

---

## Critérios de avaliação

- **Funcionalidade**: atendimento aos requisitos obrigatórios.
- **Arquitetura**: clareza, separação de responsabilidades e extensibilidade.
- **Qualidade**: organização do código, nomenclatura e boas práticas.
- **Documentação**: clareza e completude das instruções.
- **Extras**: testes, uso de frameworks de agentes e melhorias adicionais.

---

## Diferenciais

Itens que podem destacar sua entrega:

- **Arquitetura da aplicação**: diagrama em Draw.io (ou similar) ilustrando o fluxo da aplicação, componentes, agentes e integrações (Chroma, LLM, API).
- **Testes**: cobertura com testes unitários e/ou de integração para os principais fluxos (ingestão, orquestração, agentes especialistas).
- **Documentação técnica**: além do README, documentação detalhada das decisões de design e do fluxo de roteamento.

---

## Prazo sugerido

**2 dias** a partir do recebimento do código.

---

## Dúvidas

Em caso de dúvidas sobre o desafio, entre em contato com o recrutador ou com o time técnico responsável pela avaliação.
