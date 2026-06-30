"""Logger estruturado (JSONL) para histórico de re-treino."""
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("logs/retrain_history.jsonl")


def log_retrain_event(
    triggered: bool,
    deployed: bool,
    accuracy: float,
    reason: str,
) -> None:
    """Adiciona linha JSON ao log estruturado (formato JSONL)."""
    LOG_FILE.parent.mkdir(exist_ok=True)

    event = {
        "timestamp": datetime.now().isoformat(),
        "triggered": triggered,
        "deployed": deployed,
        "accuracy": accuracy,
        "reason": reason,
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(event) + "\n")


def summary() -> None:
    """Resumo simples do histórico."""
    if not LOG_FILE.exists():
        print("Sem histórico ainda.")
        return

    events = [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line]
    total = len(events)
    deployed = sum(1 for e in events if e["deployed"])

    print(f"Total execuções: {total}")
    if total:
        print(f"Deploys: {deployed} ({100 * deployed / total:.0f}%)")
        last = events[-1]
        print(f"Última: {last['timestamp']} → {last['reason']}")
