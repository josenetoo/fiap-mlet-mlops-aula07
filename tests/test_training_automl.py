"""Testes do AutoML."""
from retrain.training import train_challenger_automl


def test_automl_retorna_modelo_e_params():
    result = train_challenger_automl("2.0", n_iter=4, cv=3)
    assert 0.0 <= result.accuracy <= 1.0
    assert "n_estimators" in result.params
