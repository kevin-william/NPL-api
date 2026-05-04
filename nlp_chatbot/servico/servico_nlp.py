import logging
from typing import List

from modelos.requisicoes import DocumentoComFonte as DocReq
from modelos.respostas import (
    DocumentoComFonte,
    RespostaBusca,
    RespostaStatus,
    RespostaTreinamento,
    RespostaStopwords,
    RespostaTokenizacao,
)
from motor.motor_base import MotorNLP, ModoTokenizacao
from repositorio.repositorio_base import RepositorioAbstrato

logger = logging.getLogger(__name__)


class ServicoNLP:
    """Serviço de orquestração das operações NLP."""

    def __init__(self, motor: MotorNLP, repositorio: RepositorioAbstrato, backend_armazenamento: str = "memoria"):
        self._motor = motor
        self._repositorio = repositorio
        self._backend_armazenamento = backend_armazenamento

    async def realizar_treinamento(
        self,
        documentos: List[DocReq],
        modo_tokenizacao=None,
    ) -> RespostaTreinamento:
        """Treina o modelo com os documentos fornecidos."""
        if modo_tokenizacao:
            await self._motor.definir_modo_tokenizacao(ModoTokenizacao(modo_tokenizacao))

        docs_convertidos = [DocumentoComFonte(texto=d.texto, fonte=d.fonte) for d in documentos]

        # Persiste documentos e treina
        await self._repositorio.salvar_documentos(docs_convertidos)
        await self._motor.treinar_com_documentos(docs_convertidos)

        estado = await self._motor.obter_estado_atual()
        return RespostaTreinamento(
            mensagem=f"Modelo treinado com sucesso com {len(documentos)} documento(s).",
            total_documentos=len(documentos),
            modo_tokenizacao=estado["modo_tokenizacao"],
        )

    async def executar_busca(self, consulta: str, quantidade_maxima: int = 5) -> RespostaBusca:
        """Executa busca no corpus treinado."""
        resultados = await self._motor.buscar_termos(consulta, quantidade_maxima)
        return RespostaBusca(resultados=resultados)

    async def adicionar_stopwords(self, palavras: List[str]) -> RespostaStopwords:
        """Adiciona stopwords ao motor acumulando com as existentes."""
        await self._motor.adicionar_stopwords(palavras)
        estado = await self._motor.obter_estado_atual()
        # Carrega stopwords existentes e acumula antes de persistir
        existentes = await self._repositorio.carregar_stopwords()
        todas = list(set(existentes) | set(palavras))
        await self._repositorio.salvar_stopwords(todas)
        return RespostaStopwords(
            mensagem=f"{len(palavras)} palavra(s) adicionada(s) como stopword(s).",
            total_stopwords=estado["quantidade_stopwords"],
        )

    async def obter_estatisticas(self):
        """Retorna estatísticas do corpus."""
        return await self._motor.obter_estatisticas()

    async def alterar_modo_tokenizacao(self, modo: str) -> RespostaTokenizacao:
        """Altera o modo de tokenização."""
        modo_enum = ModoTokenizacao(modo)
        await self._motor.definir_modo_tokenizacao(modo_enum)
        return RespostaTokenizacao(
            mensagem=f"Modo de tokenização alterado para '{modo}'.",
            modo_tokenizacao=modo,
        )

    async def obter_status(self) -> RespostaStatus:
        """Retorna o status atual do serviço."""
        estado = await self._motor.obter_estado_atual()
        return RespostaStatus(
            modelo_treinado=estado["modelo_treinado"],
            total_documentos=estado["total_documentos"],
            modo_tokenizacao=estado["modo_tokenizacao"],
            tamanho_vocabulario=estado["tamanho_vocabulario"],
            quantidade_stopwords=estado["quantidade_stopwords"],
            backend_armazenamento=self._backend_armazenamento,
        )
