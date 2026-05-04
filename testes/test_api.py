"""
Testes de integração para as rotas da API.
"""
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def criar_app_teste(motor_mock, repositorio_mock):
    """Cria um app FastAPI com dependências substituídas para testes."""
    from servico.servico_nlp import ServicoNLP
    from api.dependencias import obter_servico
    from api.rotas import status, treinamento, busca, estatisticas
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(status.router)
    app.include_router(treinamento.router)
    app.include_router(busca.router)
    app.include_router(estatisticas.router)

    servico = ServicoNLP(motor_mock, repositorio_mock, backend_armazenamento="memoria")
    app.dependency_overrides[obter_servico] = lambda: servico
    return app


@pytest.fixture
def cliente(motor_com_mock_spacy, repositorio_memoria):
    """Fixture que cria um TestClient com dependências mockadas."""
    app = criar_app_teste(motor_com_mock_spacy, repositorio_memoria)
    return TestClient(app)


def test_status_quando_sem_modelo_entao_retorna_nao_treinado(cliente):
    """Testa se o endpoint de status retorna não treinado quando não há modelo."""
    resposta = cliente.get("/status")

    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["modelo_treinado"] is False
    assert dados["total_documentos"] == 0
    assert dados["backend_armazenamento"] == "memoria"


def test_treinamento_quando_documentos_validos_entao_retorna_sucesso(cliente):
    """Testa se o treinamento com documentos válidos retorna sucesso."""
    payload = {
        "documentos": [
            {"texto": "Python é uma linguagem de programação.", "fonte": "doc1.txt"},
            {"texto": "FastAPI é um framework para APIs.", "fonte": "doc2.txt"},
        ]
    }
    resposta = cliente.post("/treinamento", json=payload)

    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["total_documentos"] == 2
    assert "treinado" in dados["mensagem"].lower()


def test_treinamento_com_modo_tokenizacao_personalizado(cliente):
    """Testa treinamento com modo de tokenização específico."""
    payload = {
        "documentos": [
            {"texto": "Algoritmos de machine learning.", "fonte": "ml.txt"},
        ],
        "modo_tokenizacao": "stem",
    }
    resposta = cliente.post("/treinamento", json=payload)

    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["modo_tokenizacao"] == "stem"


def test_busca_quando_modelo_treinado_entao_retorna_resultados(cliente):
    """Testa se a busca retorna resultados após treinamento."""
    # Primeiro treina
    payload_treino = {
        "documentos": [
            {"texto": "Python é excelente para ciência de dados e machine learning.", "fonte": "python.txt"},
            {"texto": "Java é uma linguagem orientada a objetos amplamente utilizada.", "fonte": "java.txt"},
        ]
    }
    cliente.post("/treinamento", json=payload_treino)

    # Depois busca
    payload_busca = {"consulta": "Python dados", "quantidade_maxima": 2}
    resposta = cliente.post("/busca", json=payload_busca)

    assert resposta.status_code == 200
    dados = resposta.json()
    assert "resultados" in dados
    assert len(dados["resultados"]) > 0
    resultado = dados["resultados"][0]
    assert "identificador_documento" in resultado
    assert "numero_paragrafo" in resultado
    assert "numero_frase" in resultado
    assert "frase_completa" in resultado
    assert "trecho_relevante" in resultado
    assert "pontuacao_similaridade" in resultado
    assert "fonte_documento" in resultado


def test_busca_quando_modelo_nao_treinado_entao_retorna_erro_400(cliente):
    """Testa se a busca sem treinamento retorna erro 400."""
    payload = {"consulta": "Python"}
    resposta = cliente.post("/busca", json=payload)

    assert resposta.status_code == 400
    assert "treinado" in resposta.json()["detail"].lower()


def test_stopwords_quando_palavras_adicionadas_entao_lista_atualizada(cliente):
    """Testa se stopwords são adicionadas corretamente via API."""
    payload = {"palavras": ["muito", "bastante", "apenas"]}
    resposta = cliente.post("/stopwords", json=payload)

    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["total_stopwords"] == 3
    assert "adicionada" in dados["mensagem"].lower()


def test_stopwords_quando_lista_vazia_entao_retorna_erro_422(cliente):
    """Testa se lista de stopwords vazia retorna erro."""
    payload = {"palavras": []}
    resposta = cliente.post("/stopwords", json=payload)

    assert resposta.status_code == 422


def test_config_tokenizacao_quando_modo_valido_entao_altera(cliente):
    """Testa se o modo de tokenização é alterado corretamente."""
    payload = {"modo": "stem"}
    resposta = cliente.post("/config/tokenizacao", json=payload)

    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["modo_tokenizacao"] == "stem"


def test_estatisticas_quando_corpus_treinado_entao_retorna_dados(cliente):
    """Testa se as estatísticas são retornadas após treinamento."""
    payload_treino = {
        "documentos": [
            {"texto": "Inteligência artificial e aprendizado de máquina.", "fonte": "ia.txt"},
            {"texto": "Redes neurais profundas para visão computacional.", "fonte": "dl.txt"},
        ]
    }
    cliente.post("/treinamento", json=payload_treino)

    resposta = cliente.get("/estatisticas")

    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["total_documentos"] == 2
    assert dados["tamanho_vocabulario"] > 0
    assert isinstance(dados["palavras_mais_frequentes"], list)


def test_estatisticas_quando_sem_treinamento_entao_retorna_zeros(cliente):
    """Testa se as estatísticas retornam zeros quando não há treinamento."""
    resposta = cliente.get("/estatisticas")

    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["total_documentos"] == 0
    assert dados["tamanho_vocabulario"] == 0


def test_treinamento_acumula_documentos_entre_requests(cliente):
    """Testa se múltiplos requests de treinamento acumulam os documentos."""
    payload_primeiro = {
        "documentos": [
            {"texto": "Python é uma linguagem de programação.", "fonte": "python.txt"},
        ]
    }
    payload_segundo = {
        "documentos": [
            {"texto": "FastAPI é um framework web assíncrono.", "fonte": "fastapi.txt"},
            {"texto": "scikit-learn fornece algoritmos de machine learning.", "fonte": "sklearn.txt"},
        ]
    }

    resposta_1 = cliente.post("/treinamento", json=payload_primeiro)
    assert resposta_1.status_code == 200
    assert resposta_1.json()["total_documentos"] == 1

    resposta_2 = cliente.post("/treinamento", json=payload_segundo)
    assert resposta_2.status_code == 200
    assert resposta_2.json()["total_documentos"] == 3


def test_treinamento_acumula_e_busca_documentos_anteriores(cliente):
    """Testa se documentos do primeiro treinamento são encontrados após segundo treinamento."""
    cliente.post("/treinamento", json={
        "documentos": [
            {"texto": "Astronomia estuda corpos celestes e o universo.", "fonte": "astro.txt"},
        ]
    })
    cliente.post("/treinamento", json={
        "documentos": [
            {"texto": "Gastronomia é a arte de cozinhar e apreciar alimentos.", "fonte": "gastro.txt"},
        ]
    })

    resposta = cliente.post("/busca", json={"consulta": "universo celeste", "quantidade_maxima": 5})

    assert resposta.status_code == 200
    fontes = [r["fonte_documento"] for r in resposta.json()["resultados"]]
    assert "astro.txt" in fontes


def test_treinamento_mensagem_informa_total_e_novos(cliente):
    """Testa se a mensagem de resposta informa total acumulado e quantidade adicionada."""
    cliente.post("/treinamento", json={
        "documentos": [
            {"texto": "Primeiro documento.", "fonte": "doc1.txt"},
        ]
    })

    resposta = cliente.post("/treinamento", json={
        "documentos": [
            {"texto": "Segundo documento.", "fonte": "doc2.txt"},
            {"texto": "Terceiro documento.", "fonte": "doc3.txt"},
        ]
    })

    dados = resposta.json()
    assert dados["total_documentos"] == 3
    assert "2" in dados["mensagem"]  # novos adicionados
    assert "3" in dados["mensagem"]  # total acumulado


def test_estatisticas_refletem_total_acumulado(cliente):
    """Testa se /estatisticas reflete o total acumulado de documentos."""
    cliente.post("/treinamento", json={
        "documentos": [
            {"texto": "Documento sobre biologia celular.", "fonte": "bio.txt"},
        ]
    })
    cliente.post("/treinamento", json={
        "documentos": [
            {"texto": "Documento sobre química orgânica.", "fonte": "quim.txt"},
            {"texto": "Documento sobre física quântica.", "fonte": "fis.txt"},
        ]
    })

    resposta = cliente.get("/estatisticas")
    assert resposta.json()["total_documentos"] == 3
