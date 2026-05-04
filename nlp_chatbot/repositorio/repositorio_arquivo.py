import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

import joblib
import aiofiles

from modelos.respostas import DocumentoComFonte
from repositorio.repositorio_base import RepositorioAbstrato


class RepositorioArquivo(RepositorioAbstrato):
    """Implementação baseada em arquivos do repositório."""

    def __init__(self, caminho_base: str = "./dados"):
        self._caminho_base = Path(caminho_base)
        self._caminho_base.mkdir(parents=True, exist_ok=True)
        self._arquivo_documentos = self._caminho_base / "documentos.json"
        self._arquivo_modelo = self._caminho_base / "modelo.joblib"
        self._arquivo_stopwords = self._caminho_base / "stopwords.json"

    async def salvar_documentos(self, documentos: List[DocumentoComFonte]) -> None:
        dados = [doc.model_dump() for doc in documentos]
        async with aiofiles.open(self._arquivo_documentos, "w", encoding="utf-8") as f:
            await f.write(json.dumps(dados, ensure_ascii=False, indent=2))

    async def carregar_documentos(self) -> List[DocumentoComFonte]:
        if not self._arquivo_documentos.exists():
            return []
        async with aiofiles.open(self._arquivo_documentos, "r", encoding="utf-8") as f:
            conteudo = await f.read()
        dados = json.loads(conteudo)
        return [DocumentoComFonte(**item) for item in dados]

    async def salvar_modelo(self, dados_modelo: dict) -> None:
        await asyncio.to_thread(joblib.dump, dados_modelo, self._arquivo_modelo)

    async def carregar_modelo(self) -> Optional[dict]:
        if not self._arquivo_modelo.exists():
            return None
        return await asyncio.to_thread(joblib.load, self._arquivo_modelo)

    async def salvar_stopwords(self, palavras: List[str]) -> None:
        async with aiofiles.open(self._arquivo_stopwords, "w", encoding="utf-8") as f:
            await f.write(json.dumps(palavras, ensure_ascii=False))

    async def carregar_stopwords(self) -> List[str]:
        if not self._arquivo_stopwords.exists():
            return []
        async with aiofiles.open(self._arquivo_stopwords, "r", encoding="utf-8") as f:
            conteudo = await f.read()
        return json.loads(conteudo)

    async def limpar_todos_os_dados(self) -> None:
        for arquivo in [self._arquivo_documentos, self._arquivo_modelo, self._arquivo_stopwords]:
            if arquivo.exists():
                arquivo.unlink()
