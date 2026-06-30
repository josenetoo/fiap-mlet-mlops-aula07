# 📋 CHEATSHEET - Aula 07: Treinamento Automático

## Re-Treino Automático

```python
def should_retrain(current_accuracy, threshold=0.85):
    """Decide se deve retreinar."""
    if current_accuracy < threshold:
        return True
    return False

def retrain_model(X_train, y_train):
    """Re-treina modelo."""
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def compare_models(old_acc, new_acc, threshold=0.02):
    """Compara modelos."""
    improvement = new_acc - old_acc
    if improvement >= threshold:
        return True  # Substituir
    return False  # Manter antigo
```

## AutoML (busca de hiperparâmetros)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", None],
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,        # orçamento de busca
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
print(search.best_params_, search.best_score_)
```

## DAG Airflow disparada por DRIFT (branch condicional)

```python
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator


def decide_branch(**context):
    # lê o drift detectado na Aula 06 e decide o ramo
    from retrain.drift_gate import decide_branch as gate
    return gate(retrain_task_id="retrain_pipeline", skip_task_id="skip_retrain")

check = BranchPythonOperator(task_id="check_drift", python_callable=decide_branch)
retrain = PythonOperator(task_id="retrain_pipeline", python_callable=retrain_pipeline)
skip = EmptyOperator(task_id="skip_retrain")
done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

check >> [retrain, skip] >> done
```

> `trigger_rule="none_failed_min_one_success"` no `done` evita que ele seja
> pulado quando um dos ramos (retrain/skip) é skipado pelo branch.

## DAG Airflow para Re-Treino

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'auto_retrain',
    default_args=default_args,
    schedule_interval='0 2 * * *',  # 2h da manhã
    catchup=False,
)

def retrain_pipeline():
    # 1. Verificar se precisa retreinar
    # 2. Retreinar se necessário
    # 3. Comparar com modelo atual
    # 4. Deploy se melhor
    pass

retrain_task = PythonOperator(
    task_id='retrain',
    python_callable=retrain_pipeline,
    dag=dag,
)
```

## Estratégias de Re-Treino

### 1. Baseado em Tempo
```python
# Retreinar a cada N dias
if days_since_last_train >= 30:
    retrain()
```

### 2. Baseado em Performance
```python
# Retreinar se accuracy < threshold
if current_accuracy < 0.85:
    retrain()
```

### 3. Baseado em Drift
```python
# Retreinar se drift detectado
from scipy.stats import ks_2samp

statistic, p_value = ks_2samp(train_data, prod_data)
if p_value < 0.05:  # Drift!
    retrain()
```

### 4. Baseado em Volume
```python
# Retreinar a cada N novos dados
if new_data_count >= 10000:
    retrain()
```

## Pipeline Completo

```python
def auto_retrain_pipeline():
    # 1. Carregar dados
    X_train, y_train = load_new_data()
    
    # 2. Carregar modelo atual
    current_model = load_production_model()
    current_acc = evaluate(current_model, X_test, y_test)
    
    # 3. Verificar se precisa retreinar
    if should_retrain(current_acc):
        # 4. Retreinar
        new_model = retrain_model(X_train, y_train)
        new_acc = evaluate(new_model, X_test, y_test)
        
        # 5. Comparar
        if compare_models(current_acc, new_acc):
            # 6. Deploy
            deploy_model(new_model)
            print("✅ Modelo atualizado!")
        else:
            print("⏭️  Modelo antigo mantido")
    else:
        print("✅ Modelo atual está bom")
```

## MLflow Model Registry

```python
import mlflow

# Registrar modelo
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="iris-classifier"
)

# Promover para produção
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="iris-classifier",
    version=2,
    stage="Production"
)

# Carregar modelo de produção
model = mlflow.pyfunc.load_model(
    "models:/iris-classifier/Production"
)
```

## Monitoramento

```python
# Logar métricas de re-treino
with mlflow.start_run():
    mlflow.log_param("retrain_date", datetime.now())
    mlflow.log_param("data_size", len(X_train))
    mlflow.log_metric("old_accuracy", old_acc)
    mlflow.log_metric("new_accuracy", new_acc)
    mlflow.log_metric("improvement", new_acc - old_acc)
```

## Boas Práticas

```python
# 1. Sempre validar antes de deploy
assert new_accuracy > 0.80, "Accuracy muito baixa!"

# 2. Manter histórico
save_model(model, f"models/backup_{datetime.now()}.pkl")

# 3. A/B Testing
# Deploy novo modelo para 10% do tráfego primeiro

# 4. Rollback automático
if production_accuracy < 0.80:
    rollback_to_previous_model()

# 5. Alertas
if retrain_failed:
    send_alert("Re-treino falhou!")
```
