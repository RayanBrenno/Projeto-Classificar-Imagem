"""
Transfer learning com ResNet18 pré-treinada.
Objetivo: 90%+ no CIFAR-10.

Estratégia:
    1. Carrega ResNet18 com pesos do ImageNet.
    2. Adapta a primeira conv pra 3x3 (CIFAR é 32x32, kernel 7 do ImageNet é grande demais).
    3. Substitui a fc final por 10 classes.
    4. Treina tudo (fine-tuning completo) com cosine annealing.

Uso:
    python scripts/04_cifar10_resnet.py --epochs 15
    # retomar a partir de um checkpoint anterior:
    python scripts/04_cifar10_resnet.py --epochs 10 --resume models/cifar10_resnet18.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from common import device, fit, count_params

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
HIST = ROOT / "history"

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def build_model(num_classes: int = 10) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # A ResNet original usa conv 7×7 com stride=2 pensando em imagens 224×224.
    # No CIFAR-10 (32×32) isso reduziria o mapa para 16×16 logo na entrada, perdendo metade do detalhe.
    # Kernel 3×3 com stride=1 preserva a resolução espacial para imagens pequenas.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # maxpool original reduziria ainda mais; Identity o remove sem alterar o restante da arquitetura

    # Substitui a cabeça de 1000 classes (ImageNet) por 10 classes (CIFAR-10).
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_tfm = transforms.Compose([
        # RandomCrop com padding "reflect" espelha as bordas da imagem ao invés de preencher com zeros,
        # evitando que o modelo aprenda que bordas pretas = borda da imagem.
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    # Validação sem augmentation: queremos medir o modelo puro, não o modelo + aleatoriedade do augment.
    val_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train = datasets.CIFAR10(DATA, train=True, download=True, transform=train_tfm)
    val = datasets.CIFAR10(DATA, train=False, download=True, transform=val_tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val, batch_size=512, shuffle=False, num_workers=2),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--resume", type=Path, default=None,
                        help="caminho de um .pt salvo: carrega os pesos e continua treinando")
    args = parser.parse_args()

    torch.manual_seed(0)  # garante reprodutibilidade: mesma seed = mesma ordem de batches
    print(f"device: {device()}")
    train_loader, val_loader = build_loaders(args.batch_size)
    model = build_model()
    if args.resume:
        if not args.resume.exists():
            raise SystemExit(f"--resume: arquivo não encontrado: {args.resume}")
        # map_location="cpu" carrega os pesos independente de onde foram salvos (GPU ou CPU).
        # weights_only=False é necessário para carregar o dict completo com metadados (val_acc, epoch).
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        print(f"retomando de {args.resume} (val_acc anterior={state.get('val_acc', float('nan')):.4f}, ep={state.get('epoch', '?')})")
    print(f"params: {count_params(model):,}")
    hist = fit(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        # CosineAnnealingLR reduz o lr suavemente de lr até ~0 ao longo das épocas,
        # evitando que o otimizador "pule" o mínimo nas épocas finais.
        scheduler_factory=lambda opt: lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs),
        name="cifar10_resnet18",
        save_to=MODELS / "cifar10_resnet18.pt",
    )
    hist.save(HIST / "cifar10_resnet18.json")
    final = hist.epochs[-1]
    print(f"final val acc: {final.val_acc:.4f}")


if __name__ == "__main__":
    main()
