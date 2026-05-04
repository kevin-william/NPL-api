from abc import ABC, abstractmethod
from typing import List, Optional
from modelos.respostas import DocumentoComFonte


class RepositorioAbstrato(ABC):
    @abstractmethod
    async def salvar_documentos(self, documentos: List[DocumentoComFonte]) -> None:
        """Salva a lista de documentos."""

    @abstractmethod
    async def carregar_documentos(self) -> List[DocumentoComFonte]:
        """Carrega e retorna a lista de documentos."""

    @abstractmethod
    async def salvar_modelo(self, dados_modelo: dict) -> None:
        """Salva os dados serializados do modelo."""

    @abstractmethod
    async def carregar_modelo(self) -> Optional[dict]:
        """Carrega e retorna os dados do modelo, ou None se não existir."""

    @abstractmethod
    async def salvar_stopwords(self, palavras: List[str]) -> None:
        """Salva a lista de stopwords personalizadas."""

    @abstractmethod
    async def carregar_stopwords(self) -> List[str]:
        """Carrega e retorna a lista de stopwords personalizadas."""

    @abstractmethod
    async def limpar_todos_os_dados(self) -> None:
        """Remove todos os dados persistidos."""
