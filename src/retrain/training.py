"""Treinamento de modelo (challenger).

    - train_challenger        → modelo fixo (baseline rápido).
    - train_challenger_automl → busca de hiperparâmetros (AutoML).
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """Resultado do treino."""

    model: Any
    accuracy: float
    version: str
    params: Optional[dict] = field(default=None)


def _load_split(random_state: int = 42):
    """Carrega o dataset e separa treino/teste de forma reprodutível."""
    X, y = load_iris(return_X_y=True)
    return train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )


def train_challenger(version: str, n_estimators: int = 100) -> TrainedModel:
    """Treina um challenger com hiperparâmetros FIXOS."""
    logger.info(f"🏋️  Treinando challenger v{version} (fixo)...")
    X_tr, X_te, y_tr, y_te = _load_split()

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_tr, y_tr)

    accuracy = float(accuracy_score(y_te, model.predict(X_te)))
    logger.info(f"   Accuracy: {accuracy:.3f}")
    return TrainedModel(model=model, accuracy=accuracy, version=version)


def train_challenger_automl(
    version: str, n_iter: int = 10, cv: int = 5, random_state: int = 42
) -> TrainedModel:
    """Treina um challenger usando AutoML (busca de hiperparâmetros)."""
    logger.info(f"🤖 AutoML: buscando melhor challenger v{version} ({n_iter} combinações)...")
    X_tr, X_te, y_tr, y_te = _load_split(random_state=random_state)

    # Espaço de busca: o "cardápio" que o AutoML vai explorar.
    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_distributions=param_dist,
        n_iter=n_iter, cv=cv, scoring="accuracy",
        random_state=random_state, n_jobs=-1,
    )
    search.fit(X_tr, y_tr)

    best = search.best_estimator_
    accuracy = float(accuracy_score(y_te, best.predict(X_te)))
    logger.info(f"   🏅 Melhores params: {search.best_params_}")
    logger.info(f"   Test accuracy: {accuracy:.3f}")
    return TrainedModel(model=best, accuracy=accuracy, version=version,
                        params=search.best_params_)
