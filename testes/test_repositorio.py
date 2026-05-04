"""
Testes para as implementações de repositório.
"""
import pytest


@pytest.fixture
def documentos_repositorio():
    """Fixture com documentos para testes de repositório."""
    from modelos.respostas import DocumentoComFonte
    return [
        DocumentoComFonte(texto="Primeiro documento de teste.", fonte="fonte1.txt"),
        DocumentoComFonte(texto="Segundo documento de teste.", fonte="fonte2.txt"),
    ]


@pytest.mark.asyncio
async def test_repositorio_memoria_salvar_e_carregar_documentos(documentos_repositorio):
    """Testa salvar e carregar documentos no repositório em memória."""
    from repositorio.repositorio_memoria import RepositorioMemoria

    repo = RepositorioMemoria()
    await repo.salvar_documentos(documentos_repositorio)
    carregados = await repo.carregar_documentos()

    assert len(carregados) == len(documentos_repositorio)
    assert carregados[0].texto == documentos_repositorio[0].texto
    assert carregados[0].fonte == documentos_repositorio[0].fonte
    assert carregados[1].texto == documentos_repositorio[1].texto


@pytest.mark.asyncio
async def test_repositorio_memoria_salvar_e_carregar_modelo():
    """Testa salvar e carregar modelo no repositório em memória."""
    from repositorio.repositorio_memoria import RepositorioMemoria

    repo = RepositorioMemoria()
    dados_modelo = {"versao": "1.0", "parametros": {"alpha": 0.1}}

    await repo.salvar_modelo(dados_modelo)
    carregado = await repo.carregar_modelo()

    assert carregado is not None
    assert carregado["versao"] == "1.0"
    assert carregado["parametros"]["alpha"] == 0.1


@pytest.mark.asyncio
async def test_repositorio_memoria_carregar_modelo_vazio():
    """Testa que repositório sem modelo retorna None."""
    from repositorio.repositorio_memoria import RepositorioMemoria

    repo = RepositorioMemoria()
    carregado = await repo.carregar_modelo()

    assert carregado is None


@pytest.mark.asyncio
async def test_repositorio_memoria_salvar_e_carregar_stopwords():
    """Testa salvar e carregar stopwords no repositório em memória."""
    from repositorio.repositorio_memoria import RepositorioMemoria

    repo = RepositorioMemoria()
    palavras = ["muito", "bastante", "apenas", "somente"]

    await repo.salvar_stopwords(palavras)
    carregadas = await repo.carregar_stopwords()

    assert len(carregadas) == len(palavras)
    assert set(carregadas) == set(palavras)


@pytest.mark.asyncio
async def test_repositorio_memoria_carregar_stopwords_vazio():
    """Testa que repositório sem stopwords retorna lista vazia."""
    from repositorio.repositorio_memoria import RepositorioMemoria

    repo = RepositorioMemoria()
    carregadas = await repo.carregar_stopwords()

    assert carregadas == []


@pytest.mark.asyncio
async def test_repositorio_memoria_limpar_todos_os_dados(documentos_repositorio):
    """Testa limpeza completa dos dados no repositório em memória."""
    from repositorio.repositorio_memoria import RepositorioMemoria

    repo = RepositorioMemoria()
    await repo.salvar_documentos(documentos_repositorio)
    await repo.salvar_modelo({"dados": "teste"})
    await repo.salvar_stopwords(["palavra1"])

    await repo.limpar_todos_os_dados()

    docs = await repo.carregar_documentos()
    modelo = await repo.carregar_modelo()
    stopwords = await repo.carregar_stopwords()

    assert docs == []
    assert modelo is None
    assert stopwords == []


@pytest.mark.asyncio
async def test_repositorio_arquivo_salvar_e_carregar_documentos(tmp_path, documentos_repositorio):
    """Testa salvar e carregar documentos no repositório de arquivo."""
    from repositorio.repositorio_arquivo import RepositorioArquivo

    repo = RepositorioArquivo(str(tmp_path / "dados"))
    await repo.salvar_documentos(documentos_repositorio)
    carregados = await repo.carregar_documentos()

    assert len(carregados) == len(documentos_repositorio)
    assert carregados[0].texto == documentos_repositorio[0].texto
    assert carregados[0].fonte == documentos_repositorio[0].fonte


@pytest.mark.asyncio
async def test_repositorio_arquivo_salvar_e_carregar_stopwords(tmp_path):
    """Testa salvar e carregar stopwords no repositório de arquivo."""
    from repositorio.repositorio_arquivo import RepositorioArquivo

    repo = RepositorioArquivo(str(tmp_path / "dados"))
    palavras = ["muito", "bastante"]
    await repo.salvar_stopwords(palavras)
    carregadas = await repo.carregar_stopwords()

    assert set(carregadas) == set(palavras)


@pytest.mark.asyncio
async def test_repositorio_arquivo_carregar_quando_arquivo_nao_existe(tmp_path):
    """Testa que repositório de arquivo retorna valores padrão quando arquivo não existe."""
    from repositorio.repositorio_arquivo import RepositorioArquivo

    repo = RepositorioArquivo(str(tmp_path / "dados_novos"))
    docs = await repo.carregar_documentos()
    modelo = await repo.carregar_modelo()
    stopwords = await repo.carregar_stopwords()

    assert docs == []
    assert modelo is None
    assert stopwords == []


@pytest.mark.asyncio
async def test_repositorio_arquivo_limpar_todos_os_dados(tmp_path, documentos_repositorio):
    """Testa limpeza completa dos dados no repositório de arquivo."""
    from repositorio.repositorio_arquivo import RepositorioArquivo

    repo = RepositorioArquivo(str(tmp_path / "dados"))
    await repo.salvar_documentos(documentos_repositorio)
    await repo.salvar_stopwords(["palavra"])

    await repo.limpar_todos_os_dados()

    docs = await repo.carregar_documentos()
    stopwords = await repo.carregar_stopwords()

    assert docs == []
    assert stopwords == []
