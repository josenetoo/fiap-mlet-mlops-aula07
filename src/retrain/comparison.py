"""Comparação entre champion e challenger."""
import logging

from retrain.training import TrainedModel

logger = logging.getLogger(__name__)


def is_challenger_better(
    champion: TrainedModel,
    challenger: TrainedModel,
    improvement_threshold: float = 0.02,
) -> bool:
    """True se o challenger é melhor que o champion + margem (default 2%)."""
    diff = challenger.accuracy - champion.accuracy
    logger.info(
        f"📊 Champion: {champion.accuracy:.3f} | "
        f"Challenger: {challenger.accuracy:.3f} | Diff: {diff:+.3f}"
    )
    if diff > improvement_threshold:
        logger.info(f"   ✅ Melhor (> {improvement_threshold})")
        return True
    logger.info(f"   ❌ Melhoria insuficiente (precisa > {improvement_threshold})")
    return False
