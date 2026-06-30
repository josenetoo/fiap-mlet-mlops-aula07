"""Demonstra o AutoML buscando o melhor challenger."""
import logging

from retrain.training import train_challenger_automl

logging.basicConfig(level=logging.INFO, format="%(message)s")

result = train_challenger_automl("automl-1.0", n_iter=8, cv=5)
print(f"\nMelhor modelo: accuracy={result.accuracy:.3f}")
print(f"Hiperparâmetros vencedores: {result.params}")
