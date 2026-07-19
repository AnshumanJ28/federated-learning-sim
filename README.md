# Federated Learning Simulation

A privacy-preserving ML pipeline simulating federated learning (FedAvg) across
multiple virtual clients, benchmarked against centralized training, with
client dropout modeling, an optional differential-privacy layer, and a
containerized serving API for the final model.

**Tech stack:** Python, PyTorch, Flower (`flwr`), MLflow, FastAPI, Docker,
Streamlit, Opacus (optional)

> Built a federated learning simulation (Flower + PyTorch) benchmarking
> FedAvg against centralized training across IID and non-IID data splits,
> with client dropout modeling and an optional differential privacy layer.
> Tracked all experiments in MLflow, visualized results in a Streamlit
> dashboard, and containerized the final model-serving API with Docker and
> docker-compose for reproducible deployment.

## Setup

```bash
pip install -r requirements.txt
```

## Run the experiments

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

## Serving the final model

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

- API: http://localhost:8000 (`/predict`, `/health`, `/model-info`)
- Dashboard: http://localhost:8501

To swap in a newly trained checkpoint without rebuilding the image, just
overwrite `serving/model_store/global_model.pt` — it's mounted as a volume.

## Repo structure

```
federated-learning-sim/
├── data/partition.py          # IID / non-IID splitting logic
├── fl/
│   ├── client.py               # Flower NumPyClient
│   ├── server.py               # FedAvg strategy + simulation driver
│   ├── model.py                 # Shared model architecture
│   └── dp_utils.py              # Differential privacy wrapper
├── baseline/centralized_train.py
├── serving/                    # FastAPI inference app
├── dashboard/app.py            # Streamlit comparison dashboard
├── Dockerfile / docker-compose.yml
```

## Stretch ideas

- Swap FedAvg for FedProx and compare convergence on non-IID data
- Add a GitHub Actions workflow to rebuild/push the Docker image on commit
- Deploy the serving API to Render or a similar free-tier host
