# Aula 07 - Treinamento Automático e Re-Treino

> **Repositório**: https://github.com/josenetoo/fiap-ml-aula07

## 🎯 Objetivo

Fechar o ciclo de vida do modelo com **decisão automática** (time/performance/drift/volume), **comparação champion/challenger**, **deploy automático com rollback** e **gatilho de re-treino por drift** no Airflow.

## 📹 Vídeos desta Aula

| Vídeo | Tema | O que você vai fazer |
|-------|------|---------------------|
| 01 | Drift e Estratégias de Re-Treino | Data/concept drift + revalidação contínua + 4 políticas + `should_retrain()` |
| 02 | Pipeline Automatizado + AutoML | AutoML (busca de hiperparâmetros), Champion/Challenger, deploy, rollback |
| 03 | Gatilho por Drift com Airflow | DAG com branch condicional disparada por drift + monitoramento JSONL |

## 🏗️ Pipeline desta Aula

```
Sinais → should_retrain? → Treina challenger → Melhor? → Deploy
                                                  │
                                                  └→ Rollback (se necessário)
```

Detalhes em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 📁 Estrutura do Repositório

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── pytest.ini
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHEATSHEET.md
│   ├── HANDS-ON-07-01.md     # Drift + Estratégias
│   ├── HANDS-ON-07-02.md     # Pipeline + AutoML + Comparação
│   └── HANDS-ON-07-03.md     # Gatilho por drift (Airflow)
├── src/
│   └── retrain/
│       ├── policy.py
│       ├── training.py
│       ├── comparison.py
│       ├── deploy.py
│       ├── pipeline.py
│       ├── job.py
│       ├── drift_gate.py
│       └── monitoring.py
├── dags/
│   └── retrain_dag.py
└── tests/
    ├── test_policy.py
    ├── test_comparison.py
    └── test_pipeline.py
```

## Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Aulas 05 e 06 concluídas | Airflow rodando + Drift detection |
| Python 3.9+ | `python3 --version` |

## � Como Usar

1. **Fork** e clone este repositório
2. `pip install -r requirements.txt`
3. Siga os hands-on em `docs/HANDS-ON-07-*.md`

## 📚 Documentação

| Vídeo | Hands-on |
|-------|----------|
| 01 - Drift e Estratégias de Re-Treino | [HANDS-ON-07-01.md](docs/HANDS-ON-07-01.md) |
| 02 - Pipeline Automatizado + AutoML | [HANDS-ON-07-02.md](docs/HANDS-ON-07-02.md) |
| 03 - Gatilho por Drift com Airflow | [HANDS-ON-07-03.md](docs/HANDS-ON-07-03.md) |

**Referência rápida**: [Cheatsheet](docs/CHEATSHEET.md)

---

**FIAP - Pós Tech Machine Learning Engineering**
