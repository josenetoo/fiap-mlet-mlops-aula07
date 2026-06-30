"""Cenário 1: Modelo saudável, sem necessidade de re-treino."""
import logging

from retrain.policy import RetrainSignals, should_retrain

logging.basicConfig(level=logging.INFO, format="%(message)s")

print("=== Cenário 1: Modelo Saudável ===\n")
signals = RetrainSignals(
    days_since_last_train=10,
    current_accuracy=0.92,
    drift_detected=False,
    new_samples=200,
)
should_retrain(signals)
