"""Cenário 3: Drift detectado (sem label em produção)."""
import logging

from retrain.policy import RetrainSignals, should_retrain

logging.basicConfig(level=logging.INFO, format="%(message)s")

print("=== Cenário 3: Drift sem Label ===\n")
signals = RetrainSignals(
    days_since_last_train=10,
    current_accuracy=0.90,    # parece ok mas...
    drift_detected=True,      # DRIFT detectado nos inputs!
    new_samples=200,
)
should_retrain(signals)
