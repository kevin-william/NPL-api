from pydantic import BaseModel
from typing import List, Optional
from motor.motor_base import ModoTokenizacao


class DocumentoComFonte(BaseModel):
    texto: str
    fonte: str


class RequisicaoTreinamento(BaseModel):
    documentos: List[DocumentoComFonte]
    modo_tokenizacao: Optional[ModoTokenizacao] = None


class RequisicaoBusca(BaseModel):
    consulta: str
    quantidade_maxima: int = 5


class RequisicaoStopwords(BaseModel):
    palavras: List[str]


class RequisicaoTokenizacao(BaseModel):
    modo: ModoTokenizacao
