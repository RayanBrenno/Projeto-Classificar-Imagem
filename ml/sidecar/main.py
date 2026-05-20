"""
Microserviço FastAPI que carrega o modelo e expõe /predict.

POST /predict (multipart com `image`) -> JSON com class, confidence, top_k.

O serviço carrega o checkpoint na inicialização. Use a env var MODEL_PATH
pra apontar pra ele (default: ml/models/cifar10_resnet18.pt). Se o
checkpoint não existir, o serviço recusa subir com erro claro.

Rode:
    cd ml
    .venv/bin/uvicorn sidecar.main:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations
import io
import logging
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

logger = logging.getLogger("sidecar")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "cifar10_resnet18.pt"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL))

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

app = FastAPI(title="image-classifier sidecar", version="0.1.0")
_model: torch.nn.Module | None = None


def _load_model() -> torch.nn.Module:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"checkpoint não encontrado em {MODEL_PATH}. "
            f"Treine com: .venv/bin/python scripts/04_cifar10_resnet.py --epochs 15"
        )

    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module
    res = import_module("04_cifar10_resnet")
    model = res.build_model()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    logger.info("modelo carregado de %s (val_acc=%.4f)", MODEL_PATH, state.get("val_acc", float("nan")))
    return model


@app.on_event("startup")
def startup() -> None:
    global _model
    _model = _load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": str(MODEL_PATH)}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"imagem inválida: {exc}") from exc

    assert _model is not None  # garantido pelo startup
    x = _transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = _model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()

    pairs = sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True)
    top_k = pairs[:3]
    cls, conf = top_k[0]
    return {
        "class": cls,
        "confidence": float(conf),
        "top_k": [[c, float(p)] for c, p in top_k],
    }
