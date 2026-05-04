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
    # Todos os resultados devem ter pontuação positiva e campos preenchidos
    for resultado in resultados:
        assert resultado.pontuacao_similaridade > 0.0
        assert resultado.trecho_relevante != ""
        assert resultado.fonte_documento != ""
        assert resultado.frase_completa != ""
        assert resultado.numero_paragrafo >= 1
        assert resultado.numero_frase >= 1


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
        "total_frases",
        "modo_tokenizacao",
        "tamanho_vocabulario",
        "quantidade_stopwords",
        "spacy_disponivel",
        "nltk_disponivel",
    }
    assert chaves_esperadas.issubset(estado.keys())


def test_dividir_texto_em_paragrafos_quando_texto_com_quebras_duplas_entao_retorna_paragrafos(
    motor_com_mock_spacy,
):
    """Testa a divisão de texto em parágrafos usando quebras de linha duplas."""
    texto = "Primeiro parágrafo.\n\nSegundo parágrafo.\n\nTerceiro parágrafo."
    paragrafos = motor_com_mock_spacy._dividir_texto_em_paragrafos(texto)

    assert len(paragrafos) == 3
    assert paragrafos[0] == "Primeiro parágrafo."
    assert paragrafos[1] == "Segundo parágrafo."
    assert paragrafos[2] == "Terceiro parágrafo."


def test_dividir_texto_em_paragrafos_quando_texto_simples_entao_retorna_um_paragrafo(
    motor_com_mock_spacy,
):
    """Testa que texto sem quebras duplas é retornado como um único parágrafo."""
    texto = "Texto simples sem parágrafos."
    paragrafos = motor_com_mock_spacy._dividir_texto_em_paragrafos(texto)

    assert len(paragrafos) == 1
    assert paragrafos[0] == texto


def test_dividir_texto_em_paragrafos_quando_texto_vazio_entao_retorna_lista_vazia(
    motor_com_mock_spacy,
):
    """Testa que texto vazio resulta em lista vazia."""
    paragrafos = motor_com_mock_spacy._dividir_texto_em_paragrafos("")
    assert paragrafos == []


def test_normalizar_frase_para_busca_quando_frase_com_maiusculas_e_espacos_extras(
    motor_com_mock_spacy,
):
    """Testa que a normalização básica converte para minúsculas e remove espaços extras."""
    frase = "  Python  É  Uma  Linguagem  "
    normalizada = motor_com_mock_spacy._normalizar_frase_para_busca(frase)

    assert normalizada == "python é uma linguagem"


def test_extrair_frases_de_paragrafo_quando_spacy_disponivel_entao_usa_sents(
    motor_com_mock_spacy,
):
    """Testa que a extração de frases usa spaCy quando disponível."""
    paragrafo = "Python é uma linguagem."
    frases = motor_com_mock_spacy._extrair_frases_de_paragrafo(paragrafo)

    # O mock retorna o parágrafo inteiro como uma frase
    assert len(frases) == 1
    assert frases[0] == paragrafo


def test_extrair_frases_de_paragrafo_quando_spacy_indisponivel_entao_usa_regex(
    motor_com_mock_spacy,
):
    """Testa o fallback por regex quando spaCy não está disponível."""
    motor_com_mock_spacy._modelo_spacy = None
    paragrafo = "Primeira frase. Segunda frase. Terceira frase."
    frases = motor_com_mock_spacy._extrair_frases_de_paragrafo(paragrafo)

    assert len(frases) == 3


@pytest.mark.asyncio
async def test_treinar_com_documentos_gera_frases_processadas(
    motor_com_mock_spacy, documentos_exemplo
):
    """Testa que o treinamento popula a lista de frases processadas."""
    await motor_com_mock_spacy.treinar_com_documentos(documentos_exemplo)

    assert len(motor_com_mock_spacy._frases_processadas) > 0
    # O mock retorna uma frase por parágrafo, logo há pelo menos len(documentos_exemplo) frases
    assert len(motor_com_mock_spacy._frases_processadas) >= len(documentos_exemplo)


@pytest.mark.asyncio
async def test_treinar_com_documentos_frases_tem_metadados_corretos(
    motor_com_mock_spacy,
):
    """Testa que as frases processadas contêm metadados corretos."""
    from modelos.respostas import DocumentoComFonte

    documentos = [
        DocumentoComFonte(texto="Primeira frase do doc.", fonte="doc_a.txt"),
        DocumentoComFonte(texto="Segunda frase do doc.", fonte="doc_b.txt"),
    ]
    await motor_com_mock_spacy.treinar_com_documentos(documentos)

    frases = motor_com_mock_spacy._frases_processadas
    assert frases[0].indice_documento == 0
    assert frases[0].fonte == "doc_a.txt"
    assert frases[0].numero_paragrafo == 1
    assert frases[0].numero_frase == 1
    assert frases[1].indice_documento == 1
    assert frases[1].fonte == "doc_b.txt"


@pytest.mark.asyncio
async def test_buscar_termos_retorna_metadados_de_paragrafo_e_frase(
    motor_com_mock_spacy,
):
    """Testa que buscar_termos retorna numero_paragrafo e numero_frase corretos."""
    from modelos.respostas import DocumentoComFonte

    documentos = [
        DocumentoComFonte(texto="Aprendizado de máquina.", fonte="ml.txt"),
    ]
    await motor_com_mock_spacy.treinar_com_documentos(documentos)

    resultados = await motor_com_mock_spacy.buscar_termos("aprendizado", quantidade_maxima=1)

    assert len(resultados) > 0
    resultado = resultados[0]
    assert resultado.numero_paragrafo >= 1
    assert resultado.numero_frase >= 1
    assert resultado.frase_completa != ""
    assert resultado.identificador_documento == 0
    assert resultado.fonte_documento == "ml.txt"


@pytest.mark.asyncio
async def test_obter_estado_atual_total_frases_apos_treinamento(
    motor_com_mock_spacy, documentos_exemplo
):
    """Testa que total_frases reflete o número de frases indexadas após treinamento."""
    await motor_com_mock_spacy.treinar_com_documentos(documentos_exemplo)

    estado = await motor_com_mock_spacy.obter_estado_atual()
    assert estado["total_frases"] == len(motor_com_mock_spacy._frases_processadas)
    assert estado["total_documentos"] == len(documentos_exemplo)
