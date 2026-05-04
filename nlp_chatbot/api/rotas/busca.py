from fastapi import APIRouter, Depends, HTTPException
from modelos.requisicoes import RequisicaoBusca, RequisicaoStopwords, RequisicaoTokenizacao
from modelos.respostas import RespostaBusca, RespostaStopwords, RespostaTokenizacao
from servico.servico_nlp import ServicoNLP
from api.dependencias import obter_servico

router = APIRouter(tags=["Busca e Configuração"])


@router.post(
    "/busca",
    response_model=RespostaBusca,
    summary="Buscar documentos relevantes",
    description="Busca documentos no corpus treinado usando similaridade TF-IDF.",
)
async def buscar_documentos(
    requisicao: RequisicaoBusca,
    servico: ServicoNLP = Depends(obter_servico),
):
    status = await servico.obter_status()
    if not status.modelo_treinado:
        raise HTTPException(
            status_code=400,
            detail="O modelo ainda não foi treinado. Use POST /treinamento primeiro.",
        )
    return await servico.executar_busca(
        consulta=requisicao.consulta,
        quantidade_maxima=requisicao.quantidade_maxima,
    )


@router.post(
    "/stopwords",
    response_model=RespostaStopwords,
    summary="Adicionar stopwords",
    description="Adiciona palavras à lista de stopwords personalizadas do modelo.",
)
async def adicionar_stopwords(
    requisicao: RequisicaoStopwords,
    servico: ServicoNLP = Depends(obter_servico),
):
    if not requisicao.palavras:
        raise HTTPException(status_code=422, detail="A lista de palavras não pode estar vazia.")
    return await servico.adicionar_stopwords(requisicao.palavras)


@router.post(
    "/config/tokenizacao",
    response_model=RespostaTokenizacao,
    summary="Alterar modo de tokenização",
    description="Altera o modo de tokenização usado no processamento de texto.",
)
async def alterar_tokenizacao(
    requisicao: RequisicaoTokenizacao,
    servico: ServicoNLP = Depends(obter_servico),
):
    return await servico.alterar_modo_tokenizacao(requisicao.modo.value)
