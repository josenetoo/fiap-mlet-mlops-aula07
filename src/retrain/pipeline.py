"""Pipeline orquestrador: should_retrain → train → compare → deploy."""
import logging
from dataclasses import dataclass

from retrain.comparison import is_challenger_better
from retrain.deploy import deploy
from retrain.policy import RetrainSignals, should_retrain
from retrain.training import (
    TrainedModel,
    train_challenger,
    train_challenger_automl,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    triggered: bool
    deployed: bool
    challenger_accuracy: float = 0.0
    reason: str = ""


def run_retrain_pipeline(
    signals: RetrainSignals,
    champion: TrainedModel,
    new_version: str,
    use_automl: bool = False,
) -> PipelineResult:
    """should_retrain → train (fixo/AutoML) → compare → deploy."""
    logger.info(f"🔄 Pipeline de re-treino (v{new_version})")

    if not should_retrain(signals):
        return PipelineResult(triggered=False, deployed=False, reason="sem sinal")

    if use_automl:
        challenger = train_challenger_automl(version=new_version)
    else:
        challenger = train_challenger(version=new_version)

    if not is_challenger_better(champion, challenger):
        return PipelineResult(
            triggered=True, deployed=False,
            challenger_accuracy=challenger.accuracy,
            reason="challenger não é significativamente melhor",
        )

    deploy(challenger)
    return PipelineResult(
        triggered=True, deployed=True,
        challenger_accuracy=challenger.accuracy, reason="promovido",
    )
