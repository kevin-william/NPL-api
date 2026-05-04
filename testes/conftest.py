"""
Fixtures e configurações compartilhadas para os testes.
"""
import sys
import os
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock

# Adiciona o diretório nlp_chatbot ao sys.path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nlp_chatbot"))


def criar_mock_token_spacy(texto: str, lema: str = None, is_stop: bool = False,
                           is_punct: bool = False, is_space: bool = False):
    """Cria um mock de token spaCy."""
    token = MagicMock()
    token.text = texto
    token.lemma_ = lema or texto
    token.is_stop = is_stop
    token.is_punct = is_punct
    token.is_space = is_space
    return token


def criar_mock_doc_spacy(texto: str):
    """Cria um mock de Doc spaCy a partir de um texto."""
    palavras = texto.lower().split()
    tokens = [criar_mock_token_spacy(p, lema=p) for p in palavras]
    doc = MagicMock()
    doc.__iter__ = MagicMock(return_value=iter(tokens))
    return doc


def criar_mock_modelo_spacy():
    """Cria um mock completo do modelo spaCy."""
    modelo = MagicMock()
    modelo.side_effect = lambda texto: criar_mock_doc_spacy(texto)
    return modelo


@pytest.fixture
def mock_spacy():
    """Fixture que fornece um modelo spaCy mockado."""
    return criar_mock_modelo_spacy()


@pytest.fixture
def motor_com_mock_spacy(mock_spacy):
    """Fixture que fornece um MotorSklearnSpacyNltk com spaCy mockado."""
    from motor.motor_sklearn_spacy_nltk import MotorSklearnSpacyNltk
    motor = MotorSklearnSpacyNltk.__new__(MotorSklearnSpacyNltk)
    motor._modelo_spacy = mock_spacy
    motor._vetorizador_tfidf = None
    motor._matriz_tfidf = None
    motor._documentos_originais = []
    motor._fontes_documentos = []
    motor._stopwords_personalizadas = set()
    motor._modelo_treinado = False
    motor._modelo_spacy_nome = "mock"

    from motor.motor_base import ModoTokenizacao
    motor._modo_tokenizacao_atual = ModoTokenizacao.LEMATIZACAO
    motor._inicializar_nltk()

    from motor.tokenizador_personalizado import TokenizadorPersonalizado
    motor._tokenizador = TokenizadorPersonalizado(
        modelo_spacy=mock_spacy,
        stemmer_nltk=motor._stemmer_nltk,
        stopwords=set(),
        modo="lematizacao",
    )
    return motor


@pytest.fixture
def repositorio_memoria():
    """Fixture que fornece um repositório em memória limpo."""
    from repositorio.repositorio_memoria import RepositorioMemoria
    return RepositorioMemoria()


@pytest.fixture
def servico_nlp(motor_com_mock_spacy, repositorio_memoria):
    """Fixture que fornece um ServicoNLP com dependências mockadas."""
    from servico.servico_nlp import ServicoNLP
    return ServicoNLP(motor_com_mock_spacy, repositorio_memoria, backend_armazenamento="memoria")


@pytest.fixture
def documentos_exemplo():
    """Fixture com documentos de exemplo para testes."""
    from modelos.respostas import DocumentoComFonte
    return [
        DocumentoComFonte(
            texto="Python é uma linguagem de programação muito popular para ciência de dados.",
            fonte="doc_python.txt",
        ),
        DocumentoComFonte(
            texto="FastAPI é um framework moderno para construir APIs com Python.",
            fonte="doc_fastapi.txt",
        ),
        DocumentoComFonte(
            texto="Machine learning envolve algoritmos que aprendem a partir de dados.",
            fonte="doc_ml.txt",
        ),
    ]
