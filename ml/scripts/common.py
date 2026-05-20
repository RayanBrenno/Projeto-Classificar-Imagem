"""Shared helpers for training loops, evaluation, and plotting."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    seconds: float


@dataclass
class RunHistory:
    name: str
    epochs: list[EpochStats] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"name": self.name, "epochs": [asdict(e) for e in self.epochs]}, f, indent=2)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @torch.no_grad() desativa o cálculo de gradientes durante a avaliação.
# Sem isso o PyTorch guardaria o grafo de computação em memória sem necessidade —
# na validação só queremos medir, não atualizar pesos.
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, dev: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        loss = criterion(logits, y)
        # Multiplica pelo tamanho do batch para acumular soma ponderada;
        # dividir por n no final dá a média real (não média de médias de batches desiguais).
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()  # argmax(1) = classe com maior score
        n += x.size(0)
    return total_loss / n, correct / n


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    dev: torch.device,
    pbar_desc: str = "train",
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in tqdm(loader, desc=pbar_desc, leave=False):
        x, y = x.to(dev), y.to(dev)
        # zero_grad antes do forward: o PyTorch acumula gradientes por padrão,
        # então sem isso os gradientes do batch anterior contaminariam o atual.
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()   # calcula o gradiente de cada peso via backpropagation
        optimizer.step()  # atualiza os pesos na direção que reduz a perda
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    scheduler_factory=None,
    name: str = "model",
    save_to: Path | None = None,
) -> RunHistory:
    dev = device()
    model.to(dev)
    criterion = nn.CrossEntropyLoss()
    # weight_decay aplica regularização L2: penaliza pesos grandes para reduzir overfitting.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler_factory(optimizer) if scheduler_factory else None

    hist = RunHistory(name=name)
    best_val = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, dev, pbar_desc=f"{name} ep{epoch}")
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, dev)
        if scheduler:
            scheduler.step()  # ajusta o learning rate conforme o agendamento (ex: cosine annealing)
        dt = time.time() - t0
        hist.epochs.append(EpochStats(epoch, tr_loss, tr_acc, vl_loss, vl_acc, dt))
        print(f"[{name}] ep {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {vl_loss:.4f}/{vl_acc:.4f} | {dt:.1f}s")
        if save_to and vl_acc > best_val:
            # Salva apenas quando a val_acc melhora — garante o melhor modelo, não o último.
            best_val = vl_acc
            save_to.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "val_acc": vl_acc, "epoch": epoch}, save_to)
    return hist


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
