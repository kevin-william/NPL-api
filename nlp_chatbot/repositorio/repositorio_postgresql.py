import json
import logging
from typing import List, Optional

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import Column, Integer, Text, String, select, delete
from sqlalchemy.orm import DeclarativeBase

from modelos.respostas import DocumentoComFonte
from repositorio.repositorio_base import RepositorioAbstrato

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DocumentoORM(Base):
    __tablename__ = "documentos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    texto = Column(Text, nullable=False)
    fonte = Column(String(500), nullable=False)


class ConfiguracaoORM(Base):
    __tablename__ = "configuracoes"

    chave = Column(String(100), primary_key=True)
    valor = Column(Text, nullable=False)


class RepositorioPostgreSQL(RepositorioAbstrato):
    """Implementação PostgreSQL do repositório usando SQLAlchemy async."""

    def __init__(self, url_postgresql: str):
        self._url = url_postgresql
        self._engine = create_async_engine(url_postgresql, echo=False)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def _inicializar_tabelas(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def salvar_documentos(self, documentos: List[DocumentoComFonte]) -> None:
        async with self._session_factory() as sessao:
            await sessao.execute(delete(DocumentoORM))
            for doc in documentos:
                sessao.add(DocumentoORM(texto=doc.texto, fonte=doc.fonte))
            await sessao.commit()

    async def carregar_documentos(self) -> List[DocumentoComFonte]:
        async with self._session_factory() as sessao:
            resultado = await sessao.execute(select(DocumentoORM))
            linhas = resultado.scalars().all()
            return [DocumentoComFonte(texto=row.texto, fonte=row.fonte) for row in linhas]

    async def salvar_modelo(self, dados_modelo: dict) -> None:
        valor_json = json.dumps(dados_modelo, ensure_ascii=False)
        async with self._session_factory() as sessao:
            existente = await sessao.get(ConfiguracaoORM, "modelo")
            if existente:
                existente.valor = valor_json
            else:
                sessao.add(ConfiguracaoORM(chave="modelo", valor=valor_json))
            await sessao.commit()

    async def carregar_modelo(self) -> Optional[dict]:
        async with self._session_factory() as sessao:
            resultado = await sessao.get(ConfiguracaoORM, "modelo")
            if resultado is None:
                return None
            return json.loads(resultado.valor)

    async def salvar_stopwords(self, palavras: List[str]) -> None:
        valor_json = json.dumps(palavras, ensure_ascii=False)
        async with self._session_factory() as sessao:
            existente = await sessao.get(ConfiguracaoORM, "stopwords")
            if existente:
                existente.valor = valor_json
            else:
                sessao.add(ConfiguracaoORM(chave="stopwords", valor=valor_json))
            await sessao.commit()

    async def carregar_stopwords(self) -> List[str]:
        async with self._session_factory() as sessao:
            resultado = await sessao.get(ConfiguracaoORM, "stopwords")
            if resultado is None:
                return []
            return json.loads(resultado.valor)

    async def limpar_todos_os_dados(self) -> None:
        async with self._session_factory() as sessao:
            await sessao.execute(delete(DocumentoORM))
            await sessao.execute(delete(ConfiguracaoORM))
            await sessao.commit()
