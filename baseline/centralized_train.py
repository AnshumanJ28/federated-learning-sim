"""
baseline/centralized_train.py

Trains the same FLModel architecture on the full pooled MNIST training set.
This is the "ceiling" reference point that federated results are compared
against in Phases 4-6.
"""

import argparse
import os
import sys
import time

import mlflow
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.partition import load_mnist  # noqa: E402
from fl.model import FLModel, evaluate, train_one_epoch  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_centralized(epochs: int = 10, batch_size: int = 64, lr: float = 0.01,
                     experiment_name: str = "federated-learning-sim"):
    train_set, test_set = load_mnist()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = FLModel().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="centralized-baseline"):
        mlflow.log_params({"mode": "centralized", "epochs": epochs, "batch_size": batch_size, "lr": lr})

        start = time.time()
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE)
            test_loss, test_acc = evaluate(model, test_loader, DEVICE)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("global_loss", test_loss, step=epoch)
            mlflow.log_metric("global_accuracy", test_acc, step=epoch)

            print(f"[epoch {epoch}/{epochs}] train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

        elapsed = time.time() - start
        mlflow.log_metric("training_time_sec", elapsed)

        os.makedirs("serving/model_store", exist_ok=True)
        checkpoint_path = os.path.join("serving", "model_store", "global_model.pt")
        torch.save(
            {"model_state_dict": model.state_dict(), "test_accuracy": test_acc, "round": epochs},
            checkpoint_path,
        )
        mlflow.log_artifact(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        return model, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    run_centralized(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
