from functools import lru_cache
from configuracoes import Configuracoes
from motor.motor_sklearn_spacy_nltk import MotorSklearnSpacyNltk
from repositorio.repositorio_memoria import RepositorioMemoria
from repositorio.repositorio_arquivo import RepositorioArquivo
from servico.servico_nlp import ServicoNLP

_servico_instancia: ServicoNLP | None = None


@lru_cache(maxsize=1)
def obter_configuracoes() -> Configuracoes:
    return Configuracoes()


def obter_servico() -> ServicoNLP:
    global _servico_instancia
    if _servico_instancia is None:
        config = obter_configuracoes()
        motor = MotorSklearnSpacyNltk(
            modelo_spacy_nome=config.modelo_spacy,
            modo_inicial=config.modo_tokenizacao_padrao,
        )
        tipo = config.tipo_armazenamento
        if tipo == "arquivo":
            repositorio = RepositorioArquivo(config.caminho_armazenamento_arquivo)
        elif tipo == "postgresql":
            from repositorio.repositorio_postgresql import RepositorioPostgreSQL
            repositorio = RepositorioPostgreSQL(config.url_postgresql)
        else:
            repositorio = RepositorioMemoria()

        _servico_instancia = ServicoNLP(motor, repositorio, backend_armazenamento=tipo)
    return _servico_instancia


def resetar_servico() -> None:
    """Reseta a instância do serviço (usado em testes)."""
    global _servico_instancia
    _servico_instancia = None
    obter_configuracoes.cache_clear()
