import asyncio
import logging
from typing import List, Optional, Set
from collections import Counter

from modelos.respostas import DocumentoComFonte, ResultadoBusca, EstatisticasCorpus
from motor.motor_base import MotorNLP, ModoTokenizacao
from motor.tokenizador_personalizado import TokenizadorPersonalizado

logger = logging.getLogger(__name__)


class MotorSklearnSpacyNltk(MotorNLP):
    """Motor NLP concreto usando scikit-learn (TF-IDF), spaCy e NLTK."""

    def __init__(self, modelo_spacy_nome: str = "pt_core_news_sm", modo_inicial: str = "lematizacao"):
        self._modelo_spacy = None
        self._stemmer_nltk = None
        self._vetorizador_tfidf = None
        self._matriz_tfidf = None
        self._documentos_originais: List[str] = []
        self._fontes_documentos: List[str] = []
        self._stopwords_personalizadas: Set[str] = set()
        self._modo_tokenizacao_atual: ModoTokenizacao = ModoTokenizacao(modo_inicial)
        self._modelo_treinado: bool = False
        self._modelo_spacy_nome = modelo_spacy_nome
        self._tokenizador: Optional[TokenizadorPersonalizado] = None
        self._inicializar_nltk()
        self._carregar_modelo_spacy(modelo_spacy_nome)
        self._inicializar_tokenizador()

    def _inicializar_nltk(self) -> None:
        """Inicializa o stemmer NLTK."""
        try:
            from nltk.stem import SnowballStemmer
            import nltk
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            self._stemmer_nltk = SnowballStemmer("portuguese")
        except Exception as e:
            logger.warning(f"Não foi possível inicializar NLTK: {e}")

    def _carregar_modelo_spacy(self, nome_modelo: str) -> None:
        """Carrega o modelo spaCy, com fallback gracioso se não instalado."""
        try:
            import spacy
            self._modelo_spacy = spacy.load(nome_modelo)
            logger.info(f"Modelo spaCy '{nome_modelo}' carregado com sucesso.")
        except OSError:
            logger.warning(f"Modelo spaCy '{nome_modelo}' não encontrado. Usando tokenização básica.")
            self._modelo_spacy = None
        except Exception as e:
            logger.warning(f"Erro ao carregar modelo spaCy: {e}")
            self._modelo_spacy = None

    def _inicializar_tokenizador(self) -> None:
        """Inicializa o tokenizador personalizado."""
        self._tokenizador = TokenizadorPersonalizado(
            modelo_spacy=self._modelo_spacy,
            stemmer_nltk=self._stemmer_nltk,
            stopwords=self._stopwords_personalizadas,
            modo=self._modo_tokenizacao_atual.value,
        )

    def _obter_stopwords_nltk(self) -> Set[str]:
        """Retorna as stopwords padrão do NLTK para português."""
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("portuguese"))
        except Exception:
            return set()

    def _treinar_sincronico(self, textos: List[str]) -> None:
        """Executa o treinamento TF-IDF de forma síncrona (para uso com asyncio.to_thread)."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        stopwords_nltk = self._obter_stopwords_nltk()
        todas_stopwords = stopwords_nltk | self._stopwords_personalizadas
        self._tokenizador.atualizar_stopwords(todas_stopwords)
        self._tokenizador.atualizar_modo(self._modo_tokenizacao_atual.value)

        self._vetorizador_tfidf = TfidfVectorizer(
            analyzer=self._tokenizador.construir_analisador(),
            max_features=10000,
            min_df=1,
        )
        self._matriz_tfidf = self._vetorizador_tfidf.fit_transform(textos)

    def _buscar_sincronico(self, consulta: str, quantidade_maxima: int) -> List[ResultadoBusca]:
        """Executa a busca TF-IDF de forma síncrona."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        vetor_consulta = self._vetorizador_tfidf.transform([consulta])
        similaridades = cosine_similarity(vetor_consulta, self._matriz_tfidf).flatten()

        indices_ordenados = np.argsort(similaridades)[::-1]
        resultados = []

        for idx in indices_ordenados[:quantidade_maxima]:
            pontuacao = float(similaridades[idx])
            if pontuacao <= 0.0:
                continue
            texto_original = self._documentos_originais[idx]
            # Extrai trecho relevante (primeiros 200 caracteres)
            trecho = texto_original[:200].strip()
            if len(texto_original) > 200:
                trecho += "..."

            resultados.append(ResultadoBusca(
                identificador_documento=int(idx),
                trecho_relevante=trecho,
                pontuacao_similaridade=round(pontuacao, 4),
                fonte_documento=self._fontes_documentos[idx],
            ))

        return resultados

    async def treinar_com_documentos(self, documentos: List[DocumentoComFonte]) -> None:
        """Treina o modelo com uma lista de documentos."""
        self._documentos_originais = [doc.texto for doc in documentos]
        self._fontes_documentos = [doc.fonte for doc in documentos]
        await asyncio.to_thread(self._treinar_sincronico, self._documentos_originais)
        self._modelo_treinado = True
        logger.info(f"Modelo treinado com {len(documentos)} documentos.")

    async def adicionar_stopwords(self, palavras: List[str]) -> None:
        """Adiciona palavras à lista de stopwords personalizadas."""
        self._stopwords_personalizadas.update(palavras)
        if self._tokenizador:
            stopwords_nltk = self._obter_stopwords_nltk()
            todas_stopwords = stopwords_nltk | self._stopwords_personalizadas
            self._tokenizador.atualizar_stopwords(todas_stopwords)
        logger.info(f"Adicionadas {len(palavras)} stopwords. Total: {len(self._stopwords_personalizadas)}")

    async def buscar_termos(self, consulta: str, quantidade_maxima: int = 5) -> List[ResultadoBusca]:
        """Busca documentos relevantes para a consulta."""
        if not self._modelo_treinado or self._vetorizador_tfidf is None:
            return []
        return await asyncio.to_thread(self._buscar_sincronico, consulta, quantidade_maxima)

    async def obter_estatisticas(self, palavra_alvo: Optional[str] = None) -> EstatisticasCorpus:
        """Retorna estatísticas do corpus indexado."""
        if not self._modelo_treinado or self._vetorizador_tfidf is None:
            return EstatisticasCorpus(
                total_documentos=0,
                tamanho_vocabulario=0,
                palavras_mais_frequentes=[],
            )

        vocabulario = self._vetorizador_tfidf.vocabulary_
        tamanho_vocab = len(vocabulario)

        # Conta frequências usando o corpus de treino
        contador: Counter = Counter()
        for texto in self._documentos_originais:
            tokens = self._tokenizador.processar_texto_para_tokens(texto)
            contador.update(tokens)

        palavras_mais_frequentes = contador.most_common(10)

        return EstatisticasCorpus(
            total_documentos=len(self._documentos_originais),
            tamanho_vocabulario=tamanho_vocab,
            palavras_mais_frequentes=palavras_mais_frequentes,
        )

    async def definir_modo_tokenizacao(self, modo: ModoTokenizacao) -> None:
        """Altera o modo de tokenização."""
        self._modo_tokenizacao_atual = modo
        if self._tokenizador:
            self._tokenizador.atualizar_modo(modo.value)
        logger.info(f"Modo de tokenização alterado para: {modo.value}")

    async def obter_estado_atual(self) -> dict:
        """Retorna o estado interno atual do motor."""
        vocabulario_tamanho = 0
        if self._vetorizador_tfidf and hasattr(self._vetorizador_tfidf, "vocabulary_"):
            vocabulario_tamanho = len(self._vetorizador_tfidf.vocabulary_)

        return {
            "modelo_treinado": self._modelo_treinado,
            "total_documentos": len(self._documentos_originais),
            "modo_tokenizacao": self._modo_tokenizacao_atual.value,
            "tamanho_vocabulario": vocabulario_tamanho,
            "quantidade_stopwords": len(self._stopwords_personalizadas),
            "spacy_disponivel": self._modelo_spacy is not None,
            "nltk_disponivel": self._stemmer_nltk is not None,
        }
