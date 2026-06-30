"""Testes do gate de drift (ponte Aula 06 → re-treino)."""
import json
from pathlib import Path

from retrain.drift_gate import build_signals, decide_branch, read_drift_report
from retrain.policy import RetrainSignals


def test_read_report_inexistente_assume_sem_drift(tmp_path):
    report = read_drift_report(tmp_path / "nao_existe.json")
    assert report["drift_detected"] is False


def test_read_report_existente(tmp_path):
    path = tmp_path / "drift.json"
    path.write_text(json.dumps({"drift_detected": True, "drift_score": 0.3}))
    report = read_drift_report(path)
    assert report["drift_detected"] is True
    assert report["drift_score"] == 0.3


def test_build_signals_converte_report():
    report = {
        "drift_detected": True,
        "days_since_last_train": 12,
        "current_accuracy": 0.88,
        "new_samples": 1200,
    }
    signals = build_signals(report)
    assert isinstance(signals, RetrainSignals)
    assert signals.drift_detected is True
    assert signals.days_since_last_train == 12
    assert signals.new_samples == 1200


def test_branch_dispara_retrain_com_drift(monkeypatch):
    monkeypatch.setattr(
        "retrain.drift_gate.read_drift_report",
        lambda *a, **k: {"drift_detected": True, "current_accuracy": 0.90},
    )
    assert decide_branch() == "retrain_pipeline"


def test_branch_pula_sem_sinal(monkeypatch):
    monkeypatch.setattr(
        "retrain.drift_gate.read_drift_report",
        lambda *a, **k: {
            "drift_detected": False,
            "days_since_last_train": 1,
            "current_accuracy": 0.99,
            "new_samples": 10,
        },
    )
    assert decide_branch() == "skip_retrain"
