"""Demonstra rollback após deploy ruim."""
import logging

from retrain.deploy import deploy, load_production_model, rollback
from retrain.training import train_challenger

logging.basicConfig(level=logging.INFO, format="%(message)s")

# 1. Deploy v1.0
v1 = train_challenger("1.0", n_estimators=100)
deploy(v1)

# 2. Deploy v2.0 (que vamos "achar" ruim depois)
v2 = train_challenger("2.0", n_estimators=10)
deploy(v2)

print("\nProdução atual antes do rollback: v2.0")

# 3. Rollback para v1.0
rollback()
print("Após rollback: v1.0 (modelo arquivado restaurado)")

# 4. Verificar
model = load_production_model()
print(f"\nModelo carregado tem {model.n_estimators} árvores (v1.0 = 100)")
