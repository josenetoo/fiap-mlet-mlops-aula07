"""Job de re-treino executado pela task do Airflow.

Concorrência é garantida pelo Airflow (`max_active_runs=1`), então não há
lock manual aqui. A fonte de verdade dos sinais é o relatório de drift da
Aula 06 (lido via `drift_gate`).
"""
import logging

from retrain.drift_gate import build_signals, read_drift_report
from retrain.monitoring import log_retrain_event
from retrain.pipeline import run_retrain_pipeline
from retrain.training import train_challenger

logger = logging.getLogger(__name__)


def retrain_job():
    """Lê o drift, roda o pipeline e registra o resultado."""
    signals = build_signals(read_drift_report())
    champion = train_challenger("1.0")

    result = run_retrain_pipeline(signals, champion, new_version="2.0")
    logger.info(f"✅ Job concluído: {result}")

    log_retrain_event(
        triggered=result.triggered,
        deployed=result.deployed,
        accuracy=result.challenger_accuracy,
        reason=result.reason,
    )
    return result
