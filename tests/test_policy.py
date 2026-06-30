"""Testes das políticas de re-treino."""
from retrain.policy import (
    RetrainSignals,
    should_retrain,
    should_retrain_drift,
    should_retrain_performance,
    should_retrain_time,
    should_retrain_volume,
)


def test_time_based_dispara_apos_30_dias():
    assert should_retrain_time(31) is True
    assert should_retrain_time(29) is False


def test_performance_based_dispara_abaixo_do_threshold():
    assert should_retrain_performance(0.80) is True
    assert should_retrain_performance(0.90) is False


def test_drift_based_dispara_quando_drift():
    assert should_retrain_drift(True) is True
    assert should_retrain_drift(False) is False


def test_volume_based_dispara_apos_1000_amostras():
    assert should_retrain_volume(1500) is True
    assert should_retrain_volume(500) is False


def test_combinada_dispara_se_qualquer_um_disparar():
    signals = RetrainSignals(
        days_since_last_train=10,
        current_accuracy=0.95,
        drift_detected=True,
        new_samples=100,
    )
    assert should_retrain(signals) is True


def test_combinada_nao_dispara_se_tudo_ok():
    signals = RetrainSignals(
        days_since_last_train=10,
        current_accuracy=0.95,
        drift_detected=False,
        new_samples=100,
    )
    assert should_retrain(signals) is False
