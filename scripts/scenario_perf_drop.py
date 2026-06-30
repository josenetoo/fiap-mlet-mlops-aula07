"""Cenário 2: Performance caiu."""
import logging

from retrain.policy import RetrainSignals, should_retrain

logging.basicConfig(level=logging.INFO, format="%(message)s")

print("=== Cenário 2: Performance Caiu ===\n")
signals = RetrainSignals(
    days_since_last_train=10,
    current_accuracy=0.78,    # CAIU
    drift_detected=False,
    new_samples=200,
)
should_retrain(signals)
