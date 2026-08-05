<div align="center">

# Federated Learning Simulation

**Privacy-preserving ML pipeline — FedAvg across virtual clients, benchmarked against centralized training**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-FedAvg-1F9BEE?style=for-the-badge)](https://flower.ai)
[![Opacus](https://img.shields.io/badge/Opacus-Differential_Privacy-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://opacus.ai)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

*FedAvg Simulation · IID vs Non-IID · Client Dropout · Differential Privacy · MLflow Tracking · Dockerized Serving*

A hands-on testbed for the core trade-offs in federated systems: accuracy vs. privacy, IID vs. non-IID, full vs. partial client participation.

<br/>

[Architecture](#architecture) · [Running Experiments](#running-the-experiments) · [Results](#results) · [Setup](#setup)

---

</div>

## Table of Contents

<details>
<summary><b>Click to expand</b></summary>

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Screenshots](#screenshots)
5. [Tech Stack](#tech-stack)
6. [Setup](#setup)
7. [Running the Experiments](#running-the-experiments)
8. [Dashboard](#dashboard)
9. [Serving the Final Model](#serving-the-final-model)
10. [Project Structure](#project-structure)
11. [Results](#results)
12. [Stretch Ideas](#stretch-ideas)
13. [License](#license)

</details>

---

## Overview

This project builds a federated learning simulation (Flower + PyTorch) benchmarking **FedAvg against centralized training** across IID and non-IID data splits, with client dropout modeling and an optional differential privacy layer. All experiments are tracked in MLflow, visualized in a Streamlit dashboard, and the final model-serving API is containerized with Docker for reproducible deployment.

> [!NOTE]
> **Why federated learning matters:** multiple parties can train a shared model without pooling raw data. This repo is a hands-on testbed for the core trade-offs — accuracy vs. privacy, IID vs. non-IID distributions, full vs. partial client participation — that show up in real federated systems.

---

## Architecture

### End-to-End Pipeline

```mermaid
flowchart TB
    subgraph DATA["Data Layer"]
        RAW["Raw Dataset"]
        PART["Partitioner<br/><i>data/partition.py</i>"]
        IID["IID Split<br/>Uniform across clients"]
        NONIID["Non-IID Split<br/>Skewed label distribution"]
        RAW --> PART
        PART --> IID
        PART --> NONIID
    end

    subgraph TRAINING["Training Layer"]
        direction LR
        CENTRAL["Centralized Baseline<br/><i>baseline/centralized_train.py</i><br/>Standard PyTorch training"]
        FEDERATED["Federated Simulation<br/><i>fl/server.py + fl/client.py</i><br/>FedAvg via Flower"]
    end

    subgraph FEDDETAIL["FedAvg Internals"]
        SERVER["Server<br/>Aggregate weights"]
        C1["Client 1<br/>Local train"]
        C2["Client 2<br/>Local train"]
        CN["Client N<br/>Local train"]
        DP["Differential Privacy<br/><i>fl/dp_utils.py</i><br/>Opacus noise injection"]
        SERVER -->|"Distribute global model"| C1 & C2 & CN
        C1 & C2 & CN -->|"Return updated weights"| SERVER
        DP -.->|"Optional"| C1 & C2 & CN
    end

    subgraph TRACKING["Observability"]
        MLFLOW["MLflow<br/>Params, metrics, artifacts"]
        DASH["Streamlit Dashboard<br/><i>dashboard/app.py</i><br/>Accuracy curves side by side"]
        MLFLOW --> DASH
    end

    subgraph SERVING["Model Serving"]
        API["FastAPI<br/><i>serving/main.py</i><br/>/predict · /health · /model-info"]
        DOCKER["Docker + docker-compose<br/>API + Dashboard"]
        API --> DOCKER
    end

    DATA --> TRAINING
    FEDERATED --> FEDDETAIL
    TRAINING -->|"Log every run"| MLFLOW
    TRAINING -->|"Save checkpoint"| API

    style DATA fill:#1a1a2e,stroke:#58a6ff,stroke-width:2px,color:#eee
    style TRAINING fill:#1a1a2e,stroke:#3fb950,stroke-width:2px,color:#eee
    style FEDDETAIL fill:#1a1a2e,stroke:#d29922,stroke-width:2px,color:#eee
    style TRACKING fill:#1a1a2e,stroke:#bc8cff,stroke-width:2px,color:#eee
    style SERVING fill:#1a1a2e,stroke:#e94560,stroke-width:2px,color:#eee
```

### FedAvg Round Lifecycle

```mermaid
flowchart LR
    A["Server sends<br/>global model"] --> B["Clients selected<br/>(fraction_fit)"]
    B --> C["Each client<br/>trains locally"]
    C --> D{"Differential<br/>privacy?"}
    D -->|"Yes"| E["Opacus clips<br/>gradients + noise"]
    D -->|"No"| F["Raw weight<br/>updates"]
    E --> G["Server aggregates<br/>(FedAvg)"]
    F --> G
    G --> H["Updated global<br/>model"]
    H -->|"Next round"| A

    style A fill:#0d1117,stroke:#58a6ff,stroke-width:2px,color:#c9d1d9
    style B fill:#0d1117,stroke:#d29922,stroke-width:2px,color:#c9d1d9
    style C fill:#0d1117,stroke:#3fb950,stroke-width:2px,color:#c9d1d9
    style D fill:#0d1117,stroke:#d29922,stroke-width:2px,color:#c9d1d9
    style E fill:#0d1117,stroke:#f85149,stroke-width:2px,color:#c9d1d9
    style F fill:#0d1117,stroke:#8b949e,stroke-width:2px,color:#c9d1d9
    style G fill:#0d1117,stroke:#bc8cff,stroke-width:2px,color:#c9d1d9
    style H fill:#0d1117,stroke:#58a6ff,stroke-width:2px,color:#c9d1d9
```

### IID vs Non-IID Data Partitioning

```mermaid
flowchart LR
    subgraph IID["IID Partition"]
        direction TB
        I1["Client 1<br/>All labels, uniform"]
        I2["Client 2<br/>All labels, uniform"]
        I3["Client N<br/>All labels, uniform"]
    end

    subgraph NONIID["Non-IID Partition"]
        direction TB
        N1["Client 1<br/>Labels 0, 1 only"]
        N2["Client 2<br/>Labels 2, 3 only"]
        N3["Client N<br/>Labels 8, 9 only"]
    end

    IID ~~~|"Easy convergence<br/>high accuracy"| NONIID

    style IID fill:#161b22,stroke:#3fb950,stroke-width:2px,color:#c9d1d9
    style NONIID fill:#161b22,stroke:#f85149,stroke-width:2px,color:#c9d1d9
```

> [!IMPORTANT]
> Non-IID partitioning is where federated learning gets hard. When each client only sees a subset of labels, the local models diverge — FedAvg must reconcile conflicting gradients during aggregation. This is the most realistic simulation of real-world federated scenarios (hospitals seeing different patient populations, phones with different usage patterns).

---

## Features

| Feature | Description |
|:---|:---|
| **FedAvg simulation** | N virtual clients coordinated via [Flower](https://flower.ai/) |
| **IID vs Non-IID** | Direct comparison of uniform vs. skewed data partitioning |
| **Centralized baseline** | Quantify the federated "cost" — how much accuracy you trade for privacy |
| **Client dropout** | Simulate partial participation per round (configurable `fraction_fit`) |
| **Differential privacy** | Optional Opacus layer for gradient clipping + noise injection |
| **Experiment tracking** | Every run logged to MLflow (params + metrics + artifacts) |
| **Comparison dashboard** | Streamlit app plotting accuracy curves side by side |
| **Dockerized serving** | FastAPI endpoint for the final global model, containerized with docker-compose |

---

## Screenshots

| Dashboard | Serving API |
|:---:|:---:|
| ![Dashboard](./S1%20(1).png) | ![Serving API](./S1%20(2).png) |

---

## Tech Stack

| Layer | Tools | Purpose |
|:---|:---|:---|
| Modeling | Python, PyTorch | Neural network training |
| Federation | Flower (`flwr`) | FedAvg strategy, client/server simulation |
| Privacy | Opacus | Optional per-client differential privacy |
| Experiment tracking | MLflow | Params, metrics, artifacts per run |
| Dashboard | Streamlit | Side-by-side accuracy curve comparison |
| Serving | FastAPI + Uvicorn | REST API for the final global model |
| Deployment | Docker, docker-compose | Reproducible containerized serving |

---

## Setup

```bash
git clone https://github.com/AnshumanJ28/federated-learning-sim.git
cd federated-learning-sim
pip install -r requirements.txt
```

> [!TIP]
> Opacus (for differential privacy) is included in `requirements.txt` but entirely optional — all experiments run without it. Pass `--use-dp` only when you want to measure the privacy/accuracy trade-off.

---

## Running the Experiments

### 1. Build partitions (IID + non-IID)

```bash
python -m data.partition
```

### 2. Centralized baseline

```bash
python -m baseline.centralized_train --epochs 10
```

### 3. Federated experiments

| Experiment | Command |
|:---|:---|
| **IID, full participation** | `python -m fl.server --partition iid --rounds 20 --num-clients 10` |
| **Non-IID, full participation** | `python -m fl.server --partition non_iid --rounds 20 --num-clients 10` |
| **Non-IID, 60% dropout** | `python -m fl.server --partition non_iid --rounds 20 --fraction-fit 0.6` |
| **IID, with differential privacy** | `python -m fl.server --partition iid --rounds 20 --use-dp` |

### 4. View experiment logs

```bash
mlflow ui
```

Each run logs parameters and metrics to `./mlruns`. The MLflow UI lets you compare runs side by side.

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Reads directly from the local `mlruns/` tracking store and plots accuracy curves for every run side by side.

---

## Serving the Final Model

The centralized/federated scripts save a checkpoint to `serving/model_store/global_model.pt`.

### Local

```bash
cd serving
uvicorn main:app --reload
```

### Docker (API + Dashboard)

```bash
docker compose up --build
```

| Service | Endpoint |
|:---|:---|
| API | `http://localhost:8000` — `/predict`, `/health`, `/model-info` |
| Dashboard | `http://localhost:8501` |

> [!TIP]
> To swap in a newly trained checkpoint without rebuilding the image, just overwrite `serving/model_store/global_model.pt` — it's mounted as a volume.

---

## Project Structure

```
federated-learning-sim/
├── data/
│   └── partition.py                ← IID / non-IID splitting logic
├── fl/
│   ├── client.py                   ← Flower NumPyClient (local training)
│   ├── server.py                   ← FedAvg strategy + simulation driver
│   ├── model.py                    ← Shared model architecture
│   └── dp_utils.py                 ← Differential privacy wrapper (Opacus)
├── baseline/
│   └── centralized_train.py        ← Standard PyTorch training baseline
├── serving/
│   ├── main.py                     ← FastAPI inference endpoint
│   └── model_store/
│       └── global_model.pt         ← Saved checkpoint (volume-mounted)
├── dashboard/
│   └── app.py                      ← Streamlit comparison dashboard
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Results

| Setup | Final Accuracy | Rounds / Epochs | Notes |
|:---|:---:|:---:|:---|
| Centralized baseline | — | 10 epochs | Upper bound — no federation overhead |
| Federated, IID, full participation | — | 20 rounds | Best-case federated scenario |
| Federated, non-IID, full participation | — | 20 rounds | Tests convergence under data skew |
| Federated, non-IID, 60% dropout | — | 20 rounds | Simulates partial client availability |
| Federated, IID, with DP | — | 20 rounds | Measures privacy/accuracy trade-off |

> [!NOTE]
> Results table will be populated once final experiment runs complete. The experimental configurations above are designed to isolate each variable — IID vs non-IID, full vs partial participation, with vs without DP — so each row shows the marginal cost of one real-world constraint.

---

## Stretch Ideas

- [ ] Swap FedAvg for FedProx and compare convergence on non-IID data
- [ ] Add a GitHub Actions workflow to rebuild/push the Docker image on commit
- [ ] Deploy the serving API to Render or a similar free-tier host

---

## License

MIT — see [`LICENSE`](./LICENSE).

---

<div align="center">

### Privacy-Preserving ML

*FedAvg Simulation · IID vs Non-IID · Client Dropout · Differential Privacy · MLflow Tracking · Dockerized Serving*

**Train a shared model without sharing raw data.**

<br/>

Star this repo if you found it interesting!

---

*Made by [Anshuman](https://github.com/AnshumanJ28)*

</div>
