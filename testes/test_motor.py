"""
Testes para o motor NLP.
"""
import pytest


@pytest.mark.asyncio
async def test_treinar_com_documentos_quando_lista_valida_entao_modelo_treinado(
    motor_com_mock_spacy, documentos_exemplo
):
    """Testa se o modelo é treinado corretamente com documentos válidos."""
    await motor_com_mock_spacy.treinar_com_documentos(documentos_exemplo)

    estado = await motor_com_mock_spacy.obter_estado_atual()
    assert estado["modelo_treinado"] is True
    assert estado["total_documentos"] == len(documentos_exemplo)


@pytest.mark.asyncio
async def test_buscar_termos_quando_consulta_exata_entao_retorna_documento_com_fonte(
    motor_com_mock_spacy, documentos_exemplo
):
    """Testa se a busca retorna documentos relevantes com fonte correta."""
    await motor_com_mock_spacy.treinar_com_documentos(documentos_exemplo)

    resultados = await motor_com_mock_spacy.buscar_termos("Python programação", quantidade_maxima=3)

    assert len(resultados) > 0
    fontes = [r.fonte_documento for r in resultados]
    # O documento sobre Python deve estar entre os resultados
    assert any("python" in f.lower() for f in fontes)
    # Todos os resultados devem ter pontuação positiva
    for resultado in resultados:
        assert resultado.pontuacao_similaridade > 0.0
        assert resultado.trecho_relevante != ""
        assert resultado.fonte_documento != ""


@pytest.mark.asyncio
async def test_buscar_termos_quando_modelo_nao_treinado_entao_retorna_lista_vazia(
    motor_com_mock_spacy,
):
    """Testa se a busca retorna lista vazia quando o modelo não está treinado."""
    resultados = await motor_com_mock_spacy.buscar_termos("Python")
    assert resultados == []


@pytest.mark.asyncio
async def test_adicionar_stopwords_quando_palavras_validas_entao_lista_atualizada(
    motor_com_mock_spacy,
):
    """Testa se stopwords são adicionadas corretamente."""
    palavras = ["muito", "bastante", "apenas"]
    await motor_com_mock_spacy.adicionar_stopwords(palavras)

    estado = await motor_com_mock_spacy.obter_estado_atual()
    assert estado["quantidade_stopwords"] == len(palavras)


@pytest.mark.asyncio
async def test_adicionar_stopwords_acumula_quando_chamado_multiplas_vezes(
    motor_com_mock_spacy,
):
    """Testa se stopwords acumulam ao chamar múltiplas vezes."""
    await motor_com_mock_spacy.adicionar_stopwords(["palavra1", "palavra2"])
    await motor_com_mock_spacy.adicionar_stopwords(["palavra3"])

    estado = await motor_com_mock_spacy.obter_estado_atual()
    assert estado["quantidade_stopwords"] == 3


@pytest.mark.asyncio
async def test_obter_estatisticas_quando_corpus_treinado_entao_retorna_vocabulario(
    motor_com_mock_spacy, documentos_exemplo
):
    """Testa se as estatísticas do corpus são retornadas corretamente."""
    await motor_com_mock_spacy.treinar_com_documentos(documentos_exemplo)

    estatisticas = await motor_com_mock_spacy.obter_estatisticas()

    assert estatisticas.total_documentos == len(documentos_exemplo)
    assert estatisticas.tamanho_vocabulario > 0
    assert isinstance(estatisticas.palavras_mais_frequentes, list)


@pytest.mark.asyncio
async def test_obter_estatisticas_quando_sem_treinamento_entao_retorna_zeros(
    motor_com_mock_spacy,
):
    """Testa se as estatísticas retornam zeros quando não há treinamento."""
    estatisticas = await motor_com_mock_spacy.obter_estatisticas()

    assert estatisticas.total_documentos == 0
    assert estatisticas.tamanho_vocabulario == 0
    assert estatisticas.palavras_mais_frequentes == []


@pytest.mark.asyncio
async def test_definir_modo_tokenizacao_quando_modo_alterado_entao_estado_atualizado(
    motor_com_mock_spacy,
):
    """Testa se o modo de tokenização é alterado corretamente."""
    from motor.motor_base import ModoTokenizacao

    await motor_com_mock_spacy.definir_modo_tokenizacao(ModoTokenizacao.STEM)

    estado = await motor_com_mock_spacy.obter_estado_atual()
    assert estado["modo_tokenizacao"] == "stem"


@pytest.mark.asyncio
async def test_definir_modo_tokenizacao_todos_os_modos(motor_com_mock_spacy):
    """Testa se todos os modos de tokenização são aceitos."""
    from motor.motor_base import ModoTokenizacao

    for modo in ModoTokenizacao:
        await motor_com_mock_spacy.definir_modo_tokenizacao(modo)
        estado = await motor_com_mock_spacy.obter_estado_atual()
        assert estado["modo_tokenizacao"] == modo.value


@pytest.mark.asyncio
async def test_obter_estado_atual_retorna_dicionario_completo(motor_com_mock_spacy):
    """Testa se o estado atual contém todas as chaves esperadas."""
    estado = await motor_com_mock_spacy.obter_estado_atual()

    chaves_esperadas = {
        "modelo_treinado",
        "total_documentos",
        "modo_tokenizacao",
        "tamanho_vocabulario",
        "quantidade_stopwords",
        "spacy_disponivel",
        "nltk_disponivel",
    }
    assert chaves_esperadas.issubset(estado.keys())
