"""Políticas de decisão para re-treino."""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrainSignals:
    """Sinais que entram na decisão de re-treinar."""

    days_since_last_train: int
    current_accuracy: float
    drift_detected: bool
    new_samples: int


def should_retrain_time(days: int, max_days: int = 30) -> bool:
    """Re-treina a cada N dias."""
    decision = days >= max_days
    logger.info(f"[TIME]  {days}d / {max_days}d max → {decision}")
    return decision


def should_retrain_performance(accuracy: float, threshold: float = 0.85) -> bool:
    """Re-treina se accuracy caiu abaixo do threshold."""
    decision = accuracy < threshold
    logger.info(f"[PERF]  {accuracy:.3f} < {threshold} → {decision}")
    return decision


def should_retrain_drift(drift_detected: bool) -> bool:
    """Re-treina se drift foi detectado nos inputs."""
    logger.info(f"[DRIFT] detectado={drift_detected} → {drift_detected}")
    return drift_detected


def should_retrain_volume(new_samples: int, threshold: int = 1000) -> bool:
    """Re-treina se acumulou N novos exemplos."""
    decision = new_samples >= threshold
    logger.info(f"[VOL]   {new_samples} / {threshold} → {decision}")
    return decision


def should_retrain(signals: RetrainSignals) -> bool:
    """Combina TODAS estratégias. Re-treina se QUALQUER uma dispara (OR)."""
    decisions = [
        should_retrain_time(signals.days_since_last_train),
        should_retrain_performance(signals.current_accuracy),
        should_retrain_drift(signals.drift_detected),
        should_retrain_volume(signals.new_samples),
    ]
    final = any(decisions)
    logger.info(f"➡️  DECISÃO FINAL: {'RE-TREINAR' if final else 'manter modelo'}")
    return final
