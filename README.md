# Projeto Classificar Imagem

Classificador de imagens ponta a ponta usando transfer learning com ResNet18.

- **Modelo** — PyTorch, ResNet18 fine-tuned no CIFAR-10
- **API** — Rust + Axum (gateway + pré-processamento)
- **Sidecar** — Python + FastAPI (inferência)
- **Frontend** — React + Vite (upload + visualização do top-K)

---

## Resultado

| Modelo | Dataset | Params | Val acc |
|---|---|---:|---:|
| ResNet18 transfer learning | CIFAR-10 | 11M | **87.76%** |

---

## Arquitetura

```
React + Vite ──► Axum (Rust) ──► FastAPI (Python + PyTorch)
   :5173            :8000              :8001
```

Fluxo de uma request:
1. Frontend faz `POST /classify` (multipart com a imagem)
2. Vite proxia pra `127.0.0.1:8000`
3. API Rust valida, redimensiona pra 32×32 (Lanczos3) e reencoda PNG via `spawn_blocking`
4. Repassa via `reqwest` pro sidecar em `:8001/predict`
5. Sidecar normaliza, roda `model(x)`, softmax → top-K
6. JSON volta: `{class, confidence, top_k}` → frontend renderiza barras

Latência local típica: ~50-80ms total.

---

## Como rodar

### Pré-requisitos

- Python 3.12
- Rust 1.78+ → [rustup.rs](https://rustup.rs)
- Node 20+

### Primeira execução

```bash
chmod +x setup.sh
./setup.sh
```

O script faz tudo automaticamente:
1. Cria o ambiente virtual Python e instala as dependências
2. Baixa o CIFAR-10 (~170 MB) e treina o modelo (~5h em CPU, ~87% de acurácia)
3. Instala as dependências do frontend
4. Compila a API Rust em modo release

> Se já tiver um modelo treinado, o treino é pulado automaticamente.
> Para retomar um treino anterior:
> ```bash
> cd ml
> .venv/bin/python scripts/cifar10_resnet.py --resume models/cifar10_resnet18.pt --epochs 10
> ```

### Subir a aplicação

```bash
./start.sh
```

| Serviço | Endereço |
|---|---|
| Frontend | http://127.0.0.1:5173 |
| API Rust | http://127.0.0.1:8000 |
| Sidecar Python | http://127.0.0.1:8001 |

`Ctrl+C` encerra os 3 processos de uma vez.

