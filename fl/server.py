"""
fl/server.py

Runs a full FedAvg simulation with `flwr.simulation.start_simulation`
(virtual clients, no separate machines needed — works fine in Colab).

Supports:
  - IID / non-IID partition selection (Phase 5)
  - partial client participation / dropout via `fraction_fit` (Phase 6)
  - per-round MLflow logging of global loss/accuracy (Phases 4-6)
  - optional differential privacy on the client side (Phase 7)

Example:
    python -m fl.server --partition non_iid --rounds 20 --num-clients 10 \
        --fraction-fit 0.6 --local-epochs 1
"""

import argparse
from typing import Dict, List, Tuple

import flwr as fl
import mlflow
from flwr.common import Metrics

from fl.client import client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate per-client evaluate() metrics into a single weighted accuracy."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config_fn(server_round: int) -> Dict:
    """Per-round config sent to every client's fit()."""
    return {"lr": 0.01, "server_round": server_round}


def make_mlflow_fit_metrics_aggregation(run_name: str):
    """Wraps metric aggregation so each round's results are also logged to MLflow."""

    def aggregate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
        accs = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        agg = {
            "train_loss": sum(losses) / sum(examples),
            "train_accuracy": sum(accs) / sum(examples),
        }
        return agg

    return aggregate


def run_simulation(
    partition: str = "iid",
    num_clients: int = 10,
    rounds: int = 20,
    fraction_fit: float = 1.0,
    local_epochs: int = 1,
    use_dp: bool = False,
    experiment_name: str = "federated-learning-sim",
):
    run_label = f"federated-{partition}-frac{fraction_fit}-dp{use_dp}"

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_label):
        mlflow.log_params(
            {
                "mode": "federated",
                "partition": partition,
                "num_clients": num_clients,
                "rounds": rounds,
                "fraction_fit": fraction_fit,
                "local_epochs": local_epochs,
                "use_dp": use_dp,
            }
        )

        def client_factory(cid: str):
            return client_fn(cid, partition_name=partition, local_epochs=local_epochs, use_dp=use_dp)

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_fit_clients=max(1, int(num_clients * fraction_fit)),
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=make_mlflow_fit_metrics_aggregation(run_label),
            on_fit_config_fn=fit_config_fn,
        )

        history = fl.simulation.start_simulation(
            client_fn=client_factory,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
            client_resources={"num_cpus": 0.5, "num_gpus": 0},
        )

        # history.metrics_distributed["accuracy"] is a list of (round, value) tuples
        for round_num, acc in history.metrics_distributed.get("accuracy", []):
            mlflow.log_metric("global_accuracy", acc, step=round_num)
        for round_num, loss in history.losses_distributed:
            mlflow.log_metric("global_loss", loss, step=round_num)

        return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "non_iid"])
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="1.0 = full participation; e.g. 0.6 = 60%% dropout scenario")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--use-dp", action="store_true")
    args = parser.parse_args()

    run_simulation(
        partition=args.partition,
        num_clients=args.num_clients,
        rounds=args.rounds,
        fraction_fit=args.fraction_fit,
        local_epochs=args.local_epochs,
        use_dp=args.use_dp,
    )
