from typing import List, Optional
from modelos.respostas import DocumentoComFonte
from repositorio.repositorio_base import RepositorioAbstrato


class RepositorioMemoria(RepositorioAbstrato):
    """Implementação em memória do repositório."""

    def __init__(self):
        self._documentos: List[DocumentoComFonte] = []
        self._modelo: Optional[dict] = None
        self._stopwords: List[str] = []

    async def salvar_documentos(self, documentos: List[DocumentoComFonte]) -> None:
        self._documentos = list(documentos)

    async def carregar_documentos(self) -> List[DocumentoComFonte]:
        return list(self._documentos)

    async def salvar_modelo(self, dados_modelo: dict) -> None:
        self._modelo = dict(dados_modelo)

    async def carregar_modelo(self) -> Optional[dict]:
        return dict(self._modelo) if self._modelo is not None else None

    async def salvar_stopwords(self, palavras: List[str]) -> None:
        self._stopwords = list(palavras)

    async def carregar_stopwords(self) -> List[str]:
        return list(self._stopwords)

    async def limpar_todos_os_dados(self) -> None:
        self._documentos = []
        self._modelo = None
        self._stopwords = []
