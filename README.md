# 🔗 Federated Learning Simulation

A privacy-preserving ML pipeline that simulates **federated learning (FedAvg)**
across multiple virtual clients — benchmarked against centralized training,
with client dropout modeling, an optional differential-privacy layer, and a
containerized serving API for the final model.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-flwr-1F9BEE)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

This project builds a federated learning simulation (Flower + PyTorch)
benchmarking FedAvg against centralized training across **IID** and
**non-IID** data splits, with client dropout modeling and an optional
differential privacy layer. All experiments are tracked in MLflow,
visualized in a Streamlit dashboard, and the final model-serving API is
containerized with Docker and docker-compose for reproducible deployment.

**Why it matters:** federated learning lets multiple parties train a shared
model without pooling raw data — this repo is a hands-on testbed for the
core trade-offs (accuracy vs. privacy, IID vs. non-IID, full vs. partial
client participation) that show up in real federated systems.

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Running the Experiments](#running-the-experiments)
- [Dashboard](#dashboard)
- [Serving the Final Model](#serving-the-final-model)
- [Repo Structure](#repo-structure)
- [Results](#results)
- [Stretch Ideas](#stretch-ideas)
- [License](#license)

## Features

- 🧠 **FedAvg simulation** across N virtual clients using [Flower](https://flower.ai/)
- ⚖️ **IID vs. non-IID** data partitioning for direct comparison
- 📉 **Centralized baseline** to quantify the federated "cost"
- 🔌 **Client dropout modeling** — simulate partial participation per round
- 🔒 **Optional differential privacy** layer via Opacus
- 📊 **MLflow experiment tracking** for every run (params + metrics)
- 📈 **Streamlit dashboard** comparing accuracy curves side by side
- 🚀 **Dockerized FastAPI serving** for the final global model

## Screenshots

| Dashboard | Serving API |
|---|---|
| ![Dashboard](./S1%20(1).png) | ![Serving API](./S1%20(2).png) |

## Tech Stack

| Layer | Tools |
|---|---|
| Modeling | Python, PyTorch |
| Federation | Flower (`flwr`) |
| Privacy | Opacus (optional DP) |
| Experiment tracking | MLflow |
| Dashboard | Streamlit |
| Serving | FastAPI |
| Deployment | Docker, docker-compose |

## Setup

```bash
git clone <your-repo-url>
cd federated-learning-sim
pip install -r requirements.txt
```

## Running the Experiments

**1. Build partitions (IID + non-IID)**

```bash
python -m data.partition
```

**2. Centralized baseline**

```bash
python -m baseline.centralized_train --epochs 10
```

**3. Federated — full participation, IID**

```bash
python -m fl.server --partition iid --rounds 20 --num-clients 10
```

**4. Federated — non-IID comparison**

```bash
python -m fl.server --partition non_iid --rounds 20 --num-clients 10
```

**5. Federated — client dropout (60% participation per round)**

```bash
python -m fl.server --partition non_iid --rounds 20 --fraction-fit 0.6
```

**6. Federated — with differential privacy**

```bash
python -m fl.server --partition iid --rounds 20 --use-dp
```

Each run logs params/metrics to MLflow (`./mlruns`). View them with:

```bash
mlflow ui
```

## Dashboard

```bash
streamlit run dashboard/app.py
```

Reads directly from the local `mlruns/` tracking store and plots accuracy
curves for every run side by side.

## Serving the Final Model

The centralized/federated scripts save a checkpoint to
`serving/model_store/global_model.pt`. Serve it locally:

```bash
cd serving
uvicorn main:app --reload
```

Or via Docker (serving API + dashboard together):

```bash
docker compose up --build
```

- API: [http://localhost:8000](http://localhost:8000) (`/predict`, `/health`, `/model-info`)
- Dashboard: [http://localhost:8501](http://localhost:8501)

To swap in a newly trained checkpoint without rebuilding the image, just
overwrite `serving/model_store/global_model.pt` — it's mounted as a volume.

## Repo Structure

```
federated-learning-sim/
├── data/
│   └── partition.py            # IID / non-IID splitting logic
├── fl/
│   ├── client.py                # Flower NumPyClient
│   ├── server.py                # FedAvg strategy + simulation driver
│   ├── model.py                 # Shared model architecture
│   └── dp_utils.py              # Differential privacy wrapper
├── baseline/
│   └── centralized_train.py     # Centralized training baseline
├── serving/                     # FastAPI inference app
├── dashboard/
│   └── app.py                   # Streamlit comparison dashboard
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Results

<!-- Fill in once you have final numbers — a short table like this reads
     well and is the first thing recruiters/reviewers look for. -->

| Setup | Final Accuracy | Rounds/Epochs | Notes |
|---|---|---|---|
| Centralized baseline | — | 10 epochs | — |
| Federated, IID, full participation | — | 20 rounds | — |
| Federated, non-IID, full participation | — | 20 rounds | — |
| Federated, non-IID, 60% dropout | — | 20 rounds | — |
| Federated, IID, with DP | — | 20 rounds | — |

## Stretch Ideas

- [ ] Swap FedAvg for FedProx and compare convergence on non-IID data
- [ ] Add a GitHub Actions workflow to rebuild/push the Docker image on commit
- [ ] Deploy the serving API to Render or a similar free-tier host

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.
