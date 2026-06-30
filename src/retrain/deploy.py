"""Deploy e rollback de modelos."""
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import joblib

from retrain.training import TrainedModel

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
PRODUCTION_PATH = MODELS_DIR / "production.pkl"
ARCHIVE_DIR = MODELS_DIR / "archive"
METADATA_PATH = MODELS_DIR / "production_meta.json"


def deploy(model: TrainedModel) -> None:
    """Promove challenger para produção (arquivando o anterior)."""
    MODELS_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)

    if PRODUCTION_PATH.exists():  # archive p/ rollback futuro
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = ARCHIVE_DIR / f"champion_{ts}.pkl"
        shutil.copy(PRODUCTION_PATH, archived)
        logger.info(f"📦 Champion arquivado em {archived}")

    joblib.dump(model.model, PRODUCTION_PATH)
    METADATA_PATH.write_text(json.dumps({
        "version": model.version,
        "accuracy": model.accuracy,
        "deployed_at": datetime.now().isoformat(),
    }, indent=2))
    logger.info(f"🚀 Modelo v{model.version} promovido para produção")


def rollback() -> None:
    """Restaura o champion arquivado mais recente."""
    archived = sorted(ARCHIVE_DIR.glob("champion_*.pkl"))
    if not archived:
        raise RuntimeError("Sem modelos arquivados para rollback")
    shutil.copy(archived[-1], PRODUCTION_PATH)
    logger.warning(f"⏪ Rollback feito: restaurado {archived[-1].name}")


def load_production_model():
    """Carrega o modelo atual de produção."""
    return joblib.load(PRODUCTION_PATH) if PRODUCTION_PATH.exists() else None
