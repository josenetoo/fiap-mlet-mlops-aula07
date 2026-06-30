"""Executa pipeline com champion fraco + dados que melhoram.

Troque o `n_estimators`/`accuracy` do champion para simular os dois cenários:
    - champion fraco  → challenger promovido
    - champion forte  → challenger NÃO promovido
"""
import logging

from sklearn.ensemble import RandomForestClassifier

from retrain.pipeline import run_retrain_pipeline
from retrain.policy import RetrainSignals
from retrain.training import TrainedModel

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Champion antigo "fraco" (poucos estimators)
fake_champion = TrainedModel(
    model=RandomForestClassifier(n_estimators=5).fit([[0, 0, 0, 0]], [0]),
    accuracy=0.85,
    version="1.0",
)

# Sinais indicam re-treino (drift detectado)
signals = RetrainSignals(
    days_since_last_train=5,
    current_accuracy=0.85,
    drift_detected=True,
    new_samples=100,
)

result = run_retrain_pipeline(signals, fake_champion, new_version="2.0")
print(f"\nResultado: {result}")
