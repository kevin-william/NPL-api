from fastapi import APIRouter, Depends
from modelos.respostas import RespostaStatus
from servico.servico_nlp import ServicoNLP
from api.dependencias import obter_servico

router = APIRouter(tags=["Status"])


@router.get(
    "/status",
    response_model=RespostaStatus,
    summary="Status do serviço",
    description="Retorna o estado atual do modelo NLP, incluindo se está treinado e métricas do corpus.",
)
async def obter_status(servico: ServicoNLP = Depends(obter_servico)):
    return await servico.obter_status()
