"""Roda o job de re-treino uma vez (smoke local, sem Airflow).

Para simular drift, aponte o relatório de exemplo:
    PYTHONPATH=src DRIFT_REPORT_PATH=data/drift_report.example.json \
        python scripts/run_job.py
"""
import logging

from retrain.job import retrain_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

retrain_job()
