"""
data/partition.py

Loads MNIST and splits it across N virtual clients using either:
  - IID partitioning: shuffle + evenly distribute
  - Non-IID partitioning: Dirichlet-distribution-based label skew

Partition indices are saved to disk so every experiment (centralized,
federated-IID, federated-non-IID, dropout runs, DP runs) uses the exact
same client splits and results are directly comparable.
"""

import json
import os

import numpy as np
from torchvision import datasets, transforms

RNG_SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")
PARTITION_DIR = os.path.join(os.path.dirname(__file__), "partitions")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PARTITION_DIR, exist_ok=True)


def load_mnist():
    """Download (if needed) and return the MNIST train/test datasets."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return train_set, test_set


def iid_partition(dataset, num_clients: int, seed: int = RNG_SEED):
    """Shuffle all indices and split them evenly across clients."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = rng.permutation(n)
    shards = np.array_split(indices, num_clients)
    return [shard.tolist() for shard in shards]


def non_iid_partition(dataset, num_clients: int, alpha: float = 0.5, seed: int = RNG_SEED):
    """
    Dirichlet-distribution-based label-skewed split.

    Lower alpha -> more skew (clients see fewer classes, more heterogeneous).
    Higher alpha (e.g. >= 10) -> approaches IID.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    num_classes = int(targets.max()) + 1

    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    for c_idx in class_indices:
        rng.shuffle(c_idx)

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_size = len(class_indices[c])
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        # Convert proportions into integer split points
        split_points = (np.cumsum(proportions) * class_size).astype(int)[:-1]
        splits = np.split(class_indices[c], split_points)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices


def class_distribution(dataset, client_indices):
    """Return a {client_id: {class_label: count}} dict for MLflow logging."""
    targets = np.array(dataset.targets)
    dist = {}
    for client_id, idxs in enumerate(client_indices):
        labels, counts = np.unique(targets[idxs], return_counts=True)
        dist[client_id] = {int(l): int(c) for l, c in zip(labels, counts)}
    return dist


def save_partition(client_indices, name: str):
    path = os.path.join(PARTITION_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(client_indices, f)
    return path


def load_partition(name: str):
    path = os.path.join(PARTITION_DIR, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def build_and_save_all(num_clients: int = 10, alpha: float = 0.5):
    """Convenience entrypoint: build IID + non-IID partitions and persist them."""
    train_set, _ = load_mnist()

    iid = iid_partition(train_set, num_clients)
    non_iid = non_iid_partition(train_set, num_clients, alpha=alpha)

    save_partition(iid, "iid")
    save_partition(non_iid, "non_iid")

    return {
        "iid": class_distribution(train_set, iid),
        "non_iid": class_distribution(train_set, non_iid),
    }


if __name__ == "__main__":
    stats = build_and_save_all(num_clients=10, alpha=0.5)
    print("Saved partitions to", PARTITION_DIR)
    print(json.dumps(stats, indent=2))
