from fastapi import APIRouter, Depends
from modelos.respostas import EstatisticasCorpus
from servico.servico_nlp import ServicoNLP
from api.dependencias import obter_servico

router = APIRouter(tags=["Estatísticas"])


@router.get(
    "/estatisticas",
    response_model=EstatisticasCorpus,
    summary="Estatísticas do corpus",
    description="Retorna estatísticas do corpus indexado, incluindo vocabulário e palavras mais frequentes.",
)
async def obter_estatisticas(servico: ServicoNLP = Depends(obter_servico)):
    return await servico.obter_estatisticas()
