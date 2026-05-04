import re
from typing import List, Set, Optional


class TokenizadorPersonalizado:
    """Tokenizador que suporta lematização (spaCy), stemming (NLTK) e trigramas."""

    def __init__(
        self,
        modelo_spacy=None,
        stemmer_nltk=None,
        stopwords: Optional[Set[str]] = None,
        modo: str = "lematizacao",
    ):
        self._modelo_spacy = modelo_spacy
        self._stemmer_nltk = stemmer_nltk
        self._stopwords = stopwords or set()
        self._modo = modo

    def atualizar_stopwords(self, stopwords: Set[str]) -> None:
        self._stopwords = stopwords

    def atualizar_modo(self, modo: str) -> None:
        self._modo = modo

    def construir_analisador(self):
        """Retorna uma função callable compatível com o analisador do TfidfVectorizer."""
        def analisador(texto: str) -> List[str]:
            return self.processar_texto_para_tokens(texto)
        return analisador

    def processar_texto_para_tokens(self, texto: str) -> List[str]:
        """Processa um texto e retorna a lista de tokens conforme o modo configurado."""
        texto_limpo = self._limpar_texto(texto)
        tokens = self._tokenizar_basico(texto_limpo)
        tokens = self._remover_stopwords(tokens)

        if self._modo == "lematizacao":
            tokens = self._aplicar_lematizacao_spacy(texto_limpo, tokens)
        elif self._modo == "stem":
            tokens = self._aplicar_stemming_nltk(tokens)
        elif self._modo == "trigrama":
            tokens = self._gerar_trigramas_palavras(tokens)
        elif self._modo == "lematizacao_e_stem":
            tokens = self._aplicar_lematizacao_spacy(texto_limpo, tokens)
            tokens = self._aplicar_stemming_nltk(tokens)

        return [t for t in tokens if t.strip()]

    def _limpar_texto(self, texto: str) -> str:
        """Remove caracteres especiais e normaliza o texto.

        O texto é convertido para minúsculas antes desta etapa, portanto apenas
        variantes minúsculas dos caracteres acentuados precisam ser incluídas.
        """
        texto = texto.lower()
        # Mantém letras do alfabeto português (já em minúsculas), espaços e remove o restante.
        texto = re.sub(r"[^a-záàãâéêíóôõúüçñ\s]", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    def _tokenizar_basico(self, texto: str) -> List[str]:
        """Divide o texto em tokens básicos por espaços."""
        return texto.split()

    def _remover_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords da lista de tokens."""
        return [t for t in tokens if t not in self._stopwords and len(t) > 1]

    def _aplicar_lematizacao_spacy(self, texto_original: str, tokens_fallback: List[str]) -> List[str]:
        """Aplica lematização usando spaCy, com fallback para tokens básicos."""
        if self._modelo_spacy is None:
            return tokens_fallback
        try:
            doc = self._modelo_spacy(texto_original)
            lemas = [
                token.lemma_.lower()
                for token in doc
                if not token.is_stop and not token.is_punct and not token.is_space
                and len(token.lemma_) > 1
            ]
            return [l for l in lemas if l not in self._stopwords] if lemas else tokens_fallback
        except Exception:
            return tokens_fallback

    def _aplicar_stemming_nltk(self, tokens: List[str]) -> List[str]:
        """Aplica stemming usando NLTK SnowballStemmer."""
        if self._stemmer_nltk is None:
            return tokens
        return [self._stemmer_nltk.stem(t) for t in tokens]

    def _gerar_trigramas_palavras(self, tokens: List[str]) -> List[str]:
        """Gera trigramas de palavras a partir da lista de tokens."""
        if len(tokens) < 3:
            return tokens
        trigramas = [
            f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}"
            for i in range(len(tokens) - 2)
        ]
        return trigramas if trigramas else tokens
