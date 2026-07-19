"""
fl/dp_utils.py

Differential privacy add-on (Phase 7, stretch goal).

Wraps a client's model/optimizer/dataloader with Opacus so local training
does DP-SGD (per-sample gradient clipping + calibrated Gaussian noise).
Falls back to a manual clipping+noise implementation if Opacus isn't
installed, so the rest of the pipeline still runs.

Usage from fl/client.py:

    from fl.dp_utils import make_private, get_epsilon

    model, optimizer, dataloader, privacy_engine = make_private(
        model, optimizer, dataloader,
        target_epsilon=8.0, target_delta=1e-5, epochs=local_epochs,
    )
    ...
    eps = get_epsilon(privacy_engine, target_delta=1e-5)
"""

from typing import Optional

import torch


def _manual_dp_sgd_step(model, batch, criterion, optimizer, device, max_grad_norm, noise_multiplier):
    """
    Minimal manual DP-SGD fallback (per-batch clipping, not per-sample —
    a coarser approximation used only if Opacus is unavailable).
    """
    x, y = batch
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.normal(
                mean=0.0, std=noise_multiplier * max_grad_norm, size=p.grad.shape, device=device
            )
            p.grad.add_(noise)

    optimizer.step()
    return loss.item()


def make_private(
    model,
    optimizer,
    dataloader,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    epochs: int = 1,
    max_grad_norm: float = 1.0,
):
    """
    Attempts to wrap (model, optimizer, dataloader) with Opacus's PrivacyEngine
    for per-sample-clipped, noise-calibrated DP-SGD.

    Returns (model, optimizer, dataloader, privacy_engine_or_None).
    If Opacus isn't installed, returns the inputs unmodified plus None,
    and the caller should use `_manual_dp_sgd_step` instead for a rough
    approximation.
    """
    try:
        from opacus import PrivacyEngine

        privacy_engine = PrivacyEngine()
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )
        return model, optimizer, dataloader, privacy_engine

    except ImportError:
        print("[dp_utils] Opacus not installed — falling back to manual DP-SGD approximation.")
        return model, optimizer, dataloader, None


def get_epsilon(privacy_engine, target_delta: float = 1e-5) -> Optional[float]:
    """Return the privacy budget spent so far, or None if not using Opacus."""
    if privacy_engine is None:
        return None
    return privacy_engine.get_epsilon(delta=target_delta)


def train_one_epoch_dp(
    model,
    dataloader,
    optimizer,
    device,
    criterion=None,
    privacy_engine=None,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
):
    """
    Local training loop that is DP-aware:
      - if `privacy_engine` is set (Opacus), the wrapped optimizer already
        handles per-sample clipping + noise; this is just a normal loop.
      - if `privacy_engine` is None, uses the manual fallback step above.
    """
    criterion = criterion or torch.nn.CrossEntropyLoss()
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in dataloader:
        if privacy_engine is not None:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            correct += (out.argmax(1) == y).sum().item()
        else:
            batch_loss = _manual_dp_sgd_step(
                model, (x, y), criterion, optimizer, device, max_grad_norm, noise_multiplier
            )
            with torch.no_grad():
                out = model(x.to(device))
                correct += (out.argmax(1) == y.to(device)).sum().item()

        total_loss += batch_loss * x.size(0)
        total += x.size(0)

    return total_loss / total, correct / total
