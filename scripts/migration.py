#!/usr/bin/env python3
"""
Script de migração: insere 5 documentos em cada domínio (rh e tecnico)
via endpoint POST /documents.

Uso:
    python scripts/migration.py
    python scripts/migration.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Massa de dados: 5 documentos por domínio
DOCUMENTS_RH = [
    {
        "content": "Política de Férias: Todo colaborador tem direito a 30 dias de férias por ano, "
        "que podem ser divididos em até 3 períodos. O mínimo por período é de 10 dias. "
        "As férias devem ser agendadas com antecedência mínima de 30 dias junto ao gestor e ao RH. "
        "O pagamento das férias inclui o salário do período mais 1/3 constitucional.",
        "domain": "rh",
    },
    {
        "content": "Benefícios da empresa: Plano de saúde para o colaborador e dependentes, "
        "vale-refeição ou vale-alimentação, auxílio home office de R$ 150/mês, "
        "gympass para academias, seguro de vida em grupo e participação nos lucros (PLR) "
        "conforme desempenho da empresa no ano.",
        "domain": "rh",
    },
    {
        "content": "Processo de Onboarding: No primeiro dia, o novo colaborador recebe o kit de boas-vindas, "
        "acesso aos sistemas e credenciais. A primeira semana inclui apresentações com as áreas, "
        "treinamentos de compliance e segurança da informação. O buddy (colega de referência) "
        "acompanha o colaborador nos primeiros 30 dias. A avaliação de integração ocorre em 90 dias.",
        "domain": "rh",
    },
    {
        "content": "Regulamento Interno: A jornada de trabalho é de 8 horas diárias com 1 hora de almoço. "
        "O horário de entrada é flexível entre 7h e 10h. Home office é permitido até 3 dias por semana "
        "mediante acordo com o gestor. O uso de celular pessoal é permitido em horários de pausa. "
        "O dress code é casual, exceto em reuniões com clientes externos.",
        "domain": "rh",
    },
    {
        "content": "Política de Home Office: O trabalho remoto pode ser solicitado para até 3 dias por semana. "
        "O colaborador deve ter ambiente adequado com internet estável e privacidade para reuniões. "
        "É obrigatório participar das daily meetings e manter disponibilidade no Slack das 9h às 18h. "
        "Equipamentos como notebook e headset são fornecidos pela empresa. O auxílio home office "
        "cobre despesas com energia e internet.",
        "domain": "rh",
    },
]

DOCUMENTS_TECNICO = [
    {
        "content": "API de Pagamentos: A API aceita requisições POST no endpoint /v1/payments. "
        "O payload obrigatório inclui: amount (valor em centavos), currency (BRL ou USD), "
        "customer_id e payment_method (credit_card, pix ou boleto). A autenticação é feita "
        "via Bearer token no header Authorization. O ambiente de sandbox usa a base URL "
        "https://api.sandbox.exemplo.com e o de produção https://api.exemplo.com.",
        "domain": "tecnico",
    },
    {
        "content": "Guia de Integração: Para integrar a API, primeiro obtenha suas credenciais no painel "
        "de desenvolvedor. Use o client_id e client_secret para gerar um access_token via "
        "POST /oauth/token. O token expira em 3600 segundos. Implemente refresh token para "
        "renovação automática. Todas as requisições devem incluir o header X-Request-ID para "
        "rastreabilidade.",
        "domain": "tecnico",
    },
    {
        "content": "Arquitetura do Sistema: O sistema utiliza microserviços em containers Docker, "
        "orquestrados pelo Kubernetes. O banco de dados principal é PostgreSQL com réplicas "
        "de leitura. Redis é usado para cache e filas. A comunicação entre serviços usa "
        "mensageria RabbitMQ. O frontend é uma SPA em React que consome a API REST. "
        "Logs centralizados via ELK Stack.",
        "domain": "tecnico",
    },
    {
        "content": "Endpoints Disponíveis: GET /v1/customers lista clientes com paginação (page, limit). "
        "POST /v1/customers cria novo cliente. GET /v1/orders/{id} retorna detalhes do pedido. "
        "POST /v1/orders cria pedido e inicia fluxo de pagamento. GET /v1/webhooks lista "
        "webhooks configurados. Todos os endpoints retornam JSON e usam códigos HTTP padrão. "
        "Rate limit: 100 requisições por minuto por API key.",
        "domain": "tecnico",
    },
    {
        "content": "Autenticação e Tokens: Use OAuth 2.0 para autenticação. O fluxo client_credentials "
        "é recomendado para integrações server-to-server. Envie client_id e client_secret em "
        "POST /oauth/token com grant_type=client_credentials. O access_token retornado deve "
        "ser enviado no header: Authorization: Bearer <token>. Para APIs sensíveis, use "
        "também o header X-API-Key. Tokens comprometidos devem ser revogados no painel.",
        "domain": "tecnico",
    },
]


def post_document(base_url: str, content: str, domain: str) -> dict:
    """Envia um documento para o endpoint POST /documents."""
    url = f"{base_url.rstrip('/')}/documents"
    payload = json.dumps({"content": content, "domain": domain}).encode("utf-8")

    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def run_migration(base_url: str) -> None:
    """Executa a migração inserindo todos os documentos."""
    all_documents = DOCUMENTS_RH + DOCUMENTS_TECNICO
    success = 0
    errors = []

    print(f"Conectando em {base_url}")
    print(f"Inserindo {len(all_documents)} documentos (5 RH + 5 técnico)...\n")

    for i, doc in enumerate(all_documents, 1):
        domain = doc["domain"]
        content_preview = doc["content"][:50] + "..." if len(doc["content"]) > 50 else doc["content"]
        try:
            result = post_document(base_url, doc["content"], domain)
            success += 1
            doc_id = result.get("doc_id", "N/A")
            chunks = result.get("chunks_count", "N/A")
            print(f"  [{i:2}/{len(all_documents)}] OK | {domain:8} | doc_id={doc_id} | chunks={chunks}")
        except HTTPError as e:
            body = e.read().decode() if e.fp else str(e)
            errors.append((i, domain, f"HTTP {e.code}: {body}"))
            print(f"  [{i:2}/{len(all_documents)}] ERRO | {domain:8} | {e.code} - {body[:80]}")
        except URLError as e:
            errors.append((i, domain, str(e.reason)))
            print(f"  [{i:2}/{len(all_documents)}] ERRO | {domain:8} | {e.reason}")
        except Exception as e:
            errors.append((i, domain, str(e)))
            print(f"  [{i:2}/{len(all_documents)}] ERRO | {domain:8} | {e}")

    print(f"\n--- Resumo ---")
    print(f"Sucesso: {success}/{len(all_documents)}")

    if errors:
        print(f"Erros: {len(errors)}")
        sys.exit(1)
    else:
        print("Migração concluída com sucesso.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insere documentos de exemplo nos domínios RH e técnico via API."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="URL base da API (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    run_migration(args.base_url)


if __name__ == "__main__":
    main()
