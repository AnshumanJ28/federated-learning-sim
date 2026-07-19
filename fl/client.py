"""
fl/client.py

Flower client wrapping a single simulated device's local data shard.
Each virtual client:
  - receives global weights from the server (`set_parameters`)
  - trains locally for a few epochs on its own shard (`fit`)
  - returns updated weights + shard size (used for FedAvg's weighted average)
  - can locally evaluate the received weights (`evaluate`)

Optionally applies differential privacy to local training (Phase 7) when
`use_dp=True` is passed via client config.
"""

import argparse
import os
import sys

import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.partition import DATA_DIR, load_partition  # noqa: E402
from fl.dp_utils import get_epsilon, make_private, train_one_epoch_dp  # noqa: E402
from fl.model import FLModel, evaluate, get_parameters, set_parameters  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_client_data(client_id: int, partition_name: str, batch_size: int = 32):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    full_test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    client_indices = load_partition(partition_name)[client_id]
    train_subset = Subset(full_train, client_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    # Local eval uses a slice of the global test set for a quick sanity signal
    test_loader = DataLoader(full_test, batch_size=128, shuffle=False)

    return train_loader, test_loader, len(client_indices)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, partition_name: str, local_epochs: int = 1, use_dp: bool = False):
        self.client_id = client_id
        self.partition_name = partition_name
        self.local_epochs = local_epochs
        self.use_dp = use_dp

        self.model = FLModel().to(DEVICE)
        self.train_loader, self.test_loader, self.num_examples = load_client_data(
            client_id, partition_name
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=config.get("lr", 0.01), momentum=0.9)

        if self.use_dp:
            model, optimizer, loader, privacy_engine = make_private(
                self.model, optimizer, self.train_loader,
                target_epsilon=config.get("target_epsilon", 8.0),
                epochs=self.local_epochs,
            )
            for _ in range(self.local_epochs):
                loss, acc = train_one_epoch_dp(model, loader, optimizer, DEVICE, privacy_engine=privacy_engine)
            eps = get_epsilon(privacy_engine)
            metrics = {"train_loss": loss, "train_accuracy": acc, "epsilon": eps or -1.0}
        else:
            from fl.model import train_one_epoch
            for _ in range(self.local_epochs):
                loss, acc = train_one_epoch(self.model, self.train_loader, optimizer, DEVICE)
            metrics = {"train_loss": loss, "train_accuracy": acc}

        return get_parameters(self.model), self.num_examples, metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.test_loader, DEVICE)
        return float(loss), self.num_examples, {"accuracy": float(acc)}


def client_fn(cid: str, partition_name: str = "iid", local_epochs: int = 1, use_dp: bool = False):
    """Factory used by flwr.simulation.start_simulation to spin up client `cid`."""
    return FlowerClient(
        client_id=int(cid), partition_name=partition_name, local_epochs=local_epochs, use_dp=use_dp
    ).to_client()


if __name__ == "__main__":
    # Standalone mode: connect to a real (non-simulated) Flower server via gRPC.
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "non_iid"])
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--use-dp", action="store_true")
    args = parser.parse_args()

    client = FlowerClient(args.client_id, args.partition, args.local_epochs, args.use_dp)
    fl.client.start_client(server_address=args.server_address, client=client.to_client())
