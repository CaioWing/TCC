"""Regenera as figuras de telemetria e memória usadas no capítulo de resultados."""
from plot_memoria_voo17 import main as plot_memoria
from plot_verificacao_telemetria import main as plot_telemetria


if __name__ == "__main__":
    plot_telemetria()
    plot_memoria()
