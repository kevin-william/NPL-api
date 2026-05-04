import sys
import os

from fastapi import FastAPI
from api.rotas import status, treinamento, busca, estatisticas

app = FastAPI(
    title="NLP Chatbot API",
    description=(
        "API REST para um chatbot baseado em NLP usando TF-IDF, spaCy e NLTK. "
        "Permite treinar modelos com documentos personalizados e realizar buscas semânticas."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(status.router)
app.include_router(treinamento.router)
app.include_router(busca.router)
app.include_router(estatisticas.router)


@app.get("/", include_in_schema=False)
async def raiz():
    return {"mensagem": "NLP Chatbot API. Acesse /docs para a documentação."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
