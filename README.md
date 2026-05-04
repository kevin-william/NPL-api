# NLP Chatbot API

API REST para um chatbot baseado em NLP (Processamento de Linguagem Natural), usando TF-IDF, spaCy e NLTK. Permite treinar modelos com documentos personalizados e realizar buscas semânticas.

## Estrutura do Projeto

```
nlp_chatbot/
├── api/                        # Camada de API (FastAPI)
│   ├── dependencias.py         # Injeção de dependências
│   └── rotas/                  # Endpoints organizados por domínio
│       ├── status.py           # GET /status
│       ├── treinamento.py      # POST /treinamento
│       ├── busca.py            # POST /busca, /stopwords, /config/tokenizacao
│       └── estatisticas.py     # GET /estatisticas
├── servico/
│   └── servico_nlp.py          # Camada de serviço (orquestração)
├── motor/
│   ├── motor_base.py           # Interface abstrata do motor NLP
│   ├── motor_sklearn_spacy_nltk.py  # Implementação concreta (TF-IDF + spaCy + NLTK)
│   └── tokenizador_personalizado.py # Tokenizador com suporte a lematização, stemming e trigramas
├── repositorio/
│   ├── repositorio_base.py     # Interface abstrata do repositório
│   ├── repositorio_memoria.py  # Implementação em memória
│   ├── repositorio_arquivo.py  # Implementação baseada em arquivos JSON/joblib
│   └── repositorio_postgresql.py # Implementação PostgreSQL (SQLAlchemy async)
├── modelos/
│   ├── requisicoes.py          # Modelos Pydantic para requisições
│   └── respostas.py            # Modelos Pydantic para respostas
├── configuracoes.py            # Configurações via pydantic-settings
├── main.py                     # Ponto de entrada da aplicação
└── requirements.txt            # Dependências do projeto
testes/
├── conftest.py                 # Fixtures compartilhadas
├── test_motor.py               # Testes do motor NLP
├── test_api.py                 # Testes de integração da API
└── test_repositorio.py         # Testes dos repositórios
```

## Instalação

```bash
pip install -r nlp_chatbot/requirements.txt
python -m spacy download pt_core_news_sm
python -m nltk.downloader stopwords punkt
```

## Configuração

Copie o arquivo de exemplo e ajuste conforme necessário:

```bash
cp nlp_chatbot/.env.exemplo nlp_chatbot/.env
```

Variáveis de ambiente disponíveis:

| Variável | Padrão | Descrição |
|---|---|---|
| `TIPO_ARMAZENAMENTO` | `memoria` | Backend de armazenamento: `memoria`, `arquivo`, `postgresql` |
| `CAMINHO_ARMAZENAMENTO_ARQUIVO` | `./dados` | Diretório para armazenamento em arquivo |
| `URL_POSTGRESQL` | _(vazio)_ | URL de conexão PostgreSQL (ex: `postgresql+asyncpg://user:pass@host/db`) |
| `MODELO_SPACY` | `pt_core_news_sm` | Nome do modelo spaCy a carregar |
| `MODO_TOKENIZACAO_PADRAO` | `lematizacao` | Modo de tokenização inicial |
| `NIVEL_LOG` | `INFO` | Nível de log |

## Executando a API

```bash
cd nlp_chatbot
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse a documentação interativa (Swagger UI) em: http://localhost:8000/docs

## Endpoints

### `GET /status`
Retorna o estado atual do modelo NLP.

### `POST /treinamento`
Treina o modelo com uma lista de documentos.

```json
{
  "documentos": [
    {"texto": "Python é uma linguagem de programação.", "fonte": "doc1.txt"},
    {"texto": "FastAPI facilita a criação de APIs REST.", "fonte": "doc2.txt"}
  ],
  "modo_tokenizacao": "lematizacao"
}
```

### `POST /busca`
Busca documentos relevantes para uma consulta.

```json
{
  "consulta": "linguagem programação",
  "quantidade_maxima": 5
}
```

### `POST /stopwords`
Adiciona palavras à lista de stopwords.

```json
{
  "palavras": ["muito", "bastante", "apenas"]
}
```

### `POST /config/tokenizacao`
Altera o modo de tokenização. Modos disponíveis: `lematizacao`, `stem`, `trigrama`, `lematizacao_e_stem`.

```json
{
  "modo": "stem"
}
```

### `GET /estatisticas`
Retorna estatísticas do corpus indexado.

## Executando os Testes

```bash
python -m pytest testes/ -v
```

## Tecnologias Utilizadas

- **FastAPI** — Framework web assíncrono
- **scikit-learn** — TF-IDF e métricas de similaridade
- **spaCy** — Lematização em português
- **NLTK** — Stemming (SnowballStemmer)
- **SQLAlchemy** — ORM com suporte async para PostgreSQL
- **Pydantic** — Validação de dados e configurações
