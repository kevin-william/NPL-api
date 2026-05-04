from pydantic import BaseModel
from typing import List, Tuple, Optional


class DocumentoComFonte(BaseModel):
    texto: str
    fonte: str


class FraseProcessada(BaseModel):
    texto_normalizado: str
    texto_original: str
    fonte: str
    indice_documento: int
    numero_paragrafo: int
    numero_frase: int


class ResultadoBusca(BaseModel):
    identificador_documento: int
    numero_paragrafo: int
    numero_frase: int
    frase_completa: str
    trecho_relevante: str
    pontuacao_similaridade: float
    fonte_documento: str


class EstatisticasCorpus(BaseModel):
    total_documentos: int
    tamanho_vocabulario: int
    palavras_mais_frequentes: List[Tuple[str, int]]


class RespostaBusca(BaseModel):
    resultados: List[ResultadoBusca]


class RespostaStatus(BaseModel):
    modelo_treinado: bool
    total_documentos: int
    modo_tokenizacao: str
    tamanho_vocabulario: int
    quantidade_stopwords: int
    backend_armazenamento: str


class RespostaTreinamento(BaseModel):
    mensagem: str
    total_documentos: int
    modo_tokenizacao: str


class RespostaStopwords(BaseModel):
    mensagem: str
    total_stopwords: int


class RespostaTokenizacao(BaseModel):
    mensagem: str
    modo_tokenizacao: str
