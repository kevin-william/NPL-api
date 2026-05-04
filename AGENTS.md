# AGENTS.md — NLP Chatbot API

Instruções para agentes de IA (GitHub Copilot, Codex, etc.) que interagem com este repositório.

---

## Visão Geral

API REST assíncrona em **Python 3.12+** construída com **FastAPI**, que oferece busca semântica
por TF-IDF com tokenização via **spaCy** (lematização em pt-BR) e **NLTK** (stemming).

**Diretório raiz do projeto:** `NPL-api/`
**Pacote principal:** `nlp_chatbot/`
**Testes:** `testes/`

---

## Estrutura de Camadas

```
nlp_chatbot/
├── api/rotas/        → Endpoints FastAPI (uma rota por arquivo)
├── servico/          → Orquestração de casos de uso
├── motor/            → Lógica NLP (TF-IDF, tokenização)
├── repositorio/      → Persistência (memória, arquivo, PostgreSQL)
├── modelos/          → Schemas Pydantic (requisições e respostas)
└── configuracoes.py  → Configurações via pydantic-settings / .env
```

Siga estritamente esta separação de camadas. Não adicione lógica de negócio em rotas nem acesso
a dados em serviços.

---

## Convenções de Código

- **Idioma:** nomes de variáveis, funções, classes, comentários e mensagens em **português**.
- **Estilo:** PEP 8; sem type hints omitidos em funções públicas.
- **Async:** todas as funções que tocam I/O devem ser `async def`.
- **Pydantic v2:** use `model_config = SettingsConfigDict(...)` — não use `class Config`.
- **Injeção de dependência:** via `Depends()` do FastAPI, centralizado em `api/dependencias.py`.
- **Não** importe diretamente o repositório ou o motor dentro de rotas.

---

## Comandos Essenciais

> Execute todos os comandos a partir da raiz do repositório (`NPL-api/`).

```bash
# Instalar dependências
pip install -r nlp_chatbot/requirements.txt

# Baixar modelos (necessário apenas uma vez)
python -m spacy download pt_core_news_sm
python -m nltk.downloader stopwords punkt

# Iniciar a API em desenvolvimento
cd nlp_chatbot
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Executar todos os testes
python -m pytest testes/ -v

# Executar testes por módulo
python -m pytest testes/test_api.py -v
python -m pytest testes/test_motor.py -v
python -m pytest testes/test_repositorio.py -v
```

---

## Configuração de Ambiente

Copie e edite o arquivo de variáveis de ambiente:

```bash
# Linux/macOS
cp nlp_chatbot/.env.exemplo nlp_chatbot/.env

# Windows
copy nlp_chatbot\.env.exemplo nlp_chatbot\.env
```

| Variável | Padrão | Opções |
|---|---|---|
| `TIPO_ARMAZENAMENTO` | `memoria` | `memoria`, `arquivo`, `postgresql` |
| `CAMINHO_ARMAZENAMENTO_ARQUIVO` | `./dados` | qualquer caminho |
| `URL_POSTGRESQL` | _(vazio)_ | `postgresql+asyncpg://user:pass@host/db` |
| `MODELO_SPACY` | `pt_core_news_sm` | modelo spaCy válido |
| `MODO_TOKENIZACAO_PADRAO` | `lematizacao` | `lematizacao`, `stem`, `trigrama`, `tres_palavras`, `lematizacao_e_stem` |
| `NIVEL_LOG` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Adicionando Funcionalidades

### Nova rota

1. Crie o arquivo em `nlp_chatbot/api/rotas/<nome>.py` com um `APIRouter`.
2. Registre o router em `nlp_chatbot/main.py` com `app.include_router(...)`.
3. Adicione testes em `testes/test_api.py`.

### Novo repositório

1. Herde de `RepositorioBase` em `repositorio/repositorio_base.py`.
2. Implemente todos os métodos abstratos.
3. Registre a opção em `api/dependencias.py` e em `configuracoes.py`.

### Novo modo de tokenização

1. Adicione a lógica em `motor/tokenizador_personalizado.py`.
2. Exponha o modo via `POST /config/tokenizacao`.

---

## Testes

- Framework: **pytest** com `asyncio_mode = auto` (ver `pytest.ini`).
- O `pythonpath` está configurado para `nlp_chatbot/`, então importe diretamente (ex.: `from motor.motor_base import ...`).
- Fixtures compartilhadas ficam em `testes/conftest.py`.
- Use `httpx.AsyncClient` com `app` do FastAPI para testes de integração.
- **Não** use `unittest.TestCase` — use funções `async def test_*`.

---

## O que NÃO Fazer

- Não altere `pytest.ini` sem necessidade — o `pythonpath` é essencial.
- Não adicione lógica de negócio diretamente em rotas.
- Não use `requests` (síncrono) nos testes — use `httpx` async.
- Não faça commit de arquivos `.env` com credenciais reais.
- Não instale dependências fora do `requirements.txt` sem atualizar o arquivo.
