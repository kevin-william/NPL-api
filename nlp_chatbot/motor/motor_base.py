from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional
from modelos.respostas import DocumentoComFonte, ResultadoBusca, EstatisticasCorpus


class ModoTokenizacao(str, Enum):
    LEMATIZACAO = "lematizacao"
    STEM = "stem"
    TRIGRAMA = "trigrama"
    TRES_PALAVRAS = "tres_palavras"
    LEMATIZACAO_E_STEM = "lematizacao_e_stem"


class MotorNLP(ABC):
    @abstractmethod
    async def treinar_com_documentos(self, documentos: List[DocumentoComFonte]) -> None:
        """Treina o modelo com uma lista de documentos e suas fontes."""

    @abstractmethod
    async def adicionar_stopwords(self, palavras: List[str]) -> None:
        """Adiciona palavras à lista de stop words."""

    @abstractmethod
    async def buscar_termos(self, consulta: str, quantidade_maxima: int = 5) -> List[ResultadoBusca]:
        """Busca documentos relevantes para a consulta e retorna com snippets e fontes."""

    @abstractmethod
    async def obter_estatisticas(self, palavra_alvo: Optional[str] = None) -> EstatisticasCorpus:
        """Retorna estatísticas do corpus indexado."""

    @abstractmethod
    async def definir_modo_tokenizacao(self, modo: ModoTokenizacao) -> None:
        """Altera o modo de tokenização para processamento futuro."""

    @abstractmethod
    async def obter_estado_atual(self) -> dict:
        """Retorna o estado interno atual do motor."""
