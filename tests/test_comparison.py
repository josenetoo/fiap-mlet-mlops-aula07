"""Testes da comparação champion vs challenger."""
from sklearn.ensemble import RandomForestClassifier

from retrain.comparison import is_challenger_better
from retrain.training import TrainedModel


def _m(accuracy: float, version: str = "x") -> TrainedModel:
    return TrainedModel(
        model=RandomForestClassifier(n_estimators=5).fit([[0, 0, 0, 0]], [0]),
        accuracy=accuracy,
        version=version,
    )


def test_challenger_melhor_significativamente():
    champ = _m(0.85, "1.0")
    chal = _m(0.92, "2.0")
    assert is_challenger_better(champ, chal) is True


def test_challenger_pior():
    champ = _m(0.92, "1.0")
    chal = _m(0.85, "2.0")
    assert is_challenger_better(champ, chal) is False


def test_challenger_melhor_mas_dentro_da_margem():
    champ = _m(0.90, "1.0")
    chal = _m(0.91, "2.0")  # melhora 0.01 < 0.02 default
    assert is_challenger_better(champ, chal) is False


def test_threshold_customizado():
    champ = _m(0.90, "1.0")
    chal = _m(0.905, "2.0")
    assert is_challenger_better(champ, chal, improvement_threshold=0.001) is True
