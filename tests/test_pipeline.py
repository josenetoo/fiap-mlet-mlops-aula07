"""Testes do pipeline de re-treino."""
from sklearn.ensemble import RandomForestClassifier

from retrain.pipeline import run_retrain_pipeline
from retrain.policy import RetrainSignals
from retrain.training import TrainedModel


def _champion(accuracy=0.85):
    return TrainedModel(
        model=RandomForestClassifier(n_estimators=5).fit([[0, 0, 0, 0]], [0]),
        accuracy=accuracy, version="1.0",
    )


def test_nao_treina_se_sem_sinal():
    result = run_retrain_pipeline(RetrainSignals(5, 0.95, False, 100), _champion(), "2.0")
    assert result.triggered is False


def test_promove_challenger_melhor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_retrain_pipeline(RetrainSignals(5, 0.95, True, 100), _champion(0.50), "2.0")
    assert result.deployed is True


def test_nao_promove_challenger_pior(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_retrain_pipeline(RetrainSignals(5, 0.95, True, 100), _champion(0.99), "2.0")
    assert result.triggered is True and result.deployed is False
