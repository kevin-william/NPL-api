from fastapi import APIRouter, Depends
from modelos.requisicoes import RequisicaoTreinamento
from modelos.respostas import RespostaTreinamento
from servico.servico_nlp import ServicoNLP
from api.dependencias import obter_servico

router = APIRouter(tags=["Treinamento"])


@router.post(
    "/treinamento",
    response_model=RespostaTreinamento,
    summary="Treinar modelo",
    description="Treina o modelo NLP com uma lista de documentos e suas respectivas fontes.",
)
async def treinar_modelo(
    requisicao: RequisicaoTreinamento,
    servico: ServicoNLP = Depends(obter_servico),
):
    return await servico.realizar_treinamento(
        documentos=requisicao.documentos,
        modo_tokenizacao=requisicao.modo_tokenizacao,
    )
