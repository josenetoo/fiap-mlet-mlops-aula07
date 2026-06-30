"""DAG Airflow para re-treino automático DISPARADO POR DRIFT.

A DAG roda no schedule, checa o relatório de drift (Aula 06) e só dispara o
re-treino se os sinais justificarem (lógica condicional via branch).

Fluxo:
    check_drift ──(drift/sinais)──▶ retrain_pipeline ─┐
                └──(sem sinal)────▶ skip_retrain ──────┴─▶ done
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# Path do pacote `retrain` dentro do container Airflow.
RETRAIN_PKG_PATH = "/opt/airflow/dags/retrain"


def _check_drift_branch(**context) -> str:
    """Lê o drift e decide o próximo task_id (BranchPythonOperator)."""
    import sys
    sys.path.insert(0, RETRAIN_PKG_PATH)
    from retrain.drift_gate import decide_branch
    return decide_branch(
        retrain_task_id="retrain_pipeline",
        skip_task_id="skip_retrain",
        **context,
    )


def retrain_job_wrapper(**context):
    """Wrapper que importa e roda o pipeline (lazy import)."""
    import sys
    sys.path.insert(0, RETRAIN_PKG_PATH)
    from retrain.job import retrain_job
    retrain_job()


def alert_failure(context):
    """Callback de alerta (em prod: Slack/PagerDuty)."""
    ti = context["task_instance"]
    print(f"🚨 ALERTA: {ti.task_id} falhou em {ti.dag_id} (run {ti.run_id})")


default_args = {
    "owner": "fiap",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(hours=2),
    "on_failure_callback": alert_failure,
}

with DAG(
    dag_id="retrain_automatic",
    description="Re-treino automático disparado por detecção de drift",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",       # checa diariamente às 2h UTC
    catchup=False,
    max_active_runs=1,          # evita re-treinos paralelos
    tags=["fiap", "ml", "retrain", "drift"],
) as dag:

    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=_check_drift_branch,
    )

    retrain = PythonOperator(
        task_id="retrain_pipeline",
        python_callable=retrain_job_wrapper,
    )

    skip = EmptyOperator(task_id="skip_retrain")

    # `none_failed_min_one_success`: junta os dois ramos sem ser pulado
    # quando um deles é skipado pelo branch.
    done = EmptyOperator(
        task_id="done",
        trigger_rule="none_failed_min_one_success",
    )

    check_drift >> [retrain, skip] >> done
