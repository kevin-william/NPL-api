from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuracoes(BaseSettings):
    tipo_armazenamento: str = "memoria"  # "memoria", "arquivo", "postgresql"
    caminho_armazenamento_arquivo: str = "./dados"
    url_postgresql: str = ""
    modelo_spacy: str = "pt_core_news_sm"
    modo_tokenizacao_padrao: str = "lematizacao"
    nivel_log: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env")
