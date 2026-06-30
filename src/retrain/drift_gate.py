"""Ponte entre a detecção de drift (Aula 06) e a decisão de re-treino (Aula 07).

A Aula 06 produz um relatório de drift (ex.: KS test sobre os inputs de
produção). Aqui lemos esse relatório e o transformamos em `RetrainSignals`,
que alimentam `should_retrain()`. Assim a DAG dispara re-treino *por drift*,
não só por agendamento.
"""
import json
import logging
import os
from pathlib import Path

from retrain.policy import RetrainSignals, should_retrain

logger = logging.getLogger(__name__)

# Caminho onde a Aula 06 grava o resultado da detecção de drift.
# Em produção isso vem de um volume compartilhado / storage / XCom.
DRIFT_REPORT_PATH = Path(
    os.getenv("DRIFT_REPORT_PATH", "/opt/airflow/data/drift_report.json")
)


def read_drift_report(path: Path = DRIFT_REPORT_PATH) -> dict:
    """Lê o relatório de drift produzido pela Aula 06.

    Estrutura esperada (exemplo):
        {
            "drift_detected": true,
            "drift_score": 0.27,
            "days_since_last_train": 12,
            "current_accuracy": 0.88,
            "new_samples": 1200
        }

    Se o arquivo não existir (ex.: primeira execução), assume "sem drift"
    para não disparar re-treino indevido.
    """
    if not path.exists():
        logger.warning(f"⚠️  Relatório de drift não encontrado em {path} → assumindo sem drift")
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "days_since_last_train": 0,
            "current_accuracy": 1.0,
            "new_samples": 0,
        }

    report = json.loads(path.read_text())
    logger.info(f"📥 Relatório de drift lido: {report}")
    return report


def build_signals(report: dict) -> RetrainSignals:
    """Converte o relatório de drift em sinais de re-treino."""
    return RetrainSignals(
        days_since_last_train=int(report.get("days_since_last_train", 0)),
        current_accuracy=float(report.get("current_accuracy", 1.0)),
        drift_detected=bool(report.get("drift_detected", False)),
        new_samples=int(report.get("new_samples", 0)),
    )


def decide_branch(
    retrain_task_id: str = "retrain_pipeline",
    skip_task_id: str = "skip_retrain",
    **context,
) -> str:
    """Branch da DAG: decide se segue para re-treino ou pula.

    Usado pelo `BranchPythonOperator`. Retorna o `task_id` do próximo passo.
    Lê o drift real, monta os sinais e aplica `should_retrain()`.
    """
    report = read_drift_report()
    signals = build_signals(report)

    if should_retrain(signals):
        logger.info("➡️  Drift/sinais indicam re-treino → seguindo para pipeline")
        return retrain_task_id

    logger.info("➡️  Sem sinais suficientes → pulando re-treino")
    return skip_task_id
