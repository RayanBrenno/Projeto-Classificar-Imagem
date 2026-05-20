# Image Classifier — CIFAR-10

Classificador de imagens ponta a ponta:

- **Modelos** PyTorch (MLP, LeNet, CNN custom, ResNet18 fine-tuned)
- **API** Rust com Axum (gateway + pré-processamento)
- **Sidecar** Python com FastAPI (inferência)
- **Frontend** React + Vite (upload + visualização do top-K)

Implementação local — rode os 3 serviços e abra `http://127.0.0.1:5173`.

---

## Resultados

| Modelo | Dataset | Params | Val acc | Comentário |
|---|---|---:|---:|---|
| MLP (2 hidden, 256) | MNIST | 269k | **97.84%** | baseline, 5 epochs, ~40s CPU |
| LeNet-style CNN | MNIST | ~430k | **99.04%** | conv+BN+ReLU, 5 epochs |
| TinyVGG (4 blocos) | CIFAR-10 | ~5.5M | — | código pronto, treine com `03_cifar10_cnn.py` |
| **ResNet18 transfer learning** | CIFAR-10 | 11M | **87.76%** | conv1 adaptado pra 32×32, treinado em CPU |

---

## Arquitetura

```
React + Vite ──► Axum (Rust) ──► FastAPI (Python + PyTorch)
   :5173            :8000              :8001
```

Fluxo de uma request:
1. Frontend faz `POST /api/classify` (multipart com a imagem).
2. Vite proxia pra `127.0.0.1:8000`.
3. API Rust valida, decodifica, redimensiona pra 32×32 (Lanczos3) e reencoda PNG. Faz isso num `spawn_blocking` pra não travar o executor Tokio.
4. Repassa via `reqwest` pro sidecar em `:8001/predict`.
5. Sidecar aplica `transforms.Normalize`, roda `model(x)`, softmax → top-K.
6. JSON volta: `{class, confidence, top_k}` → frontend renderiza barras.

Latência local típica: decode+resize Rust ~5ms, HTTP ~2ms, inferência ResNet18 ~40ms ⇒ **~50-80ms total**.

---

## Como rodar

### Pré-requisitos

- Python 3.12
- Rust 1.78+ → [rustup.rs](https://rustup.rs)
- Node 20+
- `curl` (usado pelo `start.sh` pra checar se o sidecar subiu)

> O modelo treinado (`.pt`), a venv Python, o binário Rust e o `node_modules`
> **não estão no repositório** — é necessário gerá-los uma vez seguindo os passos abaixo.

---

### Passo 1 — Python: criar venv e instalar dependências

```bash
cd ml
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

---

### Passo 2 — Treinar o modelo

O CIFAR-10 é baixado automaticamente na primeira execução (~170 MB).

```bash
# ainda dentro de ml/
.venv/bin/python scripts/04_cifar10_resnet.py --epochs 15
```

> **Tempo estimado em CPU:** ~5h para 15 épocas (val acc ~87%).
> Para um resultado mais rápido e menor (menos épocas, menor acurácia):
> ```bash
> .venv/bin/python scripts/04_cifar10_resnet.py --epochs 5
> ```
> Para retomar de onde parou:
> ```bash
> .venv/bin/python scripts/04_cifar10_resnet.py --resume models/cifar10_resnet18.pt --epochs 10 --lr 5e-4
> ```

O modelo é salvo em `ml/models/cifar10_resnet18.pt`.
Se esse arquivo não existir, o sidecar recusa subir com um erro explicativo.

---

### Passo 3 — Compilar a API Rust

```bash
cd api-rust
cargo build --release --bin classify-api
# binário gerado em: api-rust/target/release/classify-api
```

---

### Passo 4 — Instalar dependências do frontend

```bash
cd frontend
npm install
```

---

### Passo 5 — Subir tudo com `./start.sh`

A partir da raiz do projeto:

```bash
chmod +x start.sh   # necessário apenas na primeira vez
./start.sh
```

O script sobe os 3 serviços em background e aguarda o sidecar ficar pronto:

| Serviço | Endereço |
|---|---|
| Frontend (React + Vite) | http://127.0.0.1:5173 |
| API Rust (gateway) | http://127.0.0.1:8000 |
| Sidecar Python (inferência) | http://127.0.0.1:8001 |

Pressione `Ctrl+C` para encerrar os 3 processos de uma vez.

---

### Testar sem o frontend

```bash
curl -F image=@/caminho/para/imagem.jpg http://localhost:8000/classify
# resposta: {"class":"cat","confidence":0.82,"top_k":[["cat",0.82],["dog",0.09],["deer",0.03]]}
```

---

## Decisões técnicas

- **Sidecar Python em vez de tch-rs/ONNX**: integrar via HTTP é mais honesto
  pra portfólio e mostra que você sabe acoplar sistemas. tch-rs tem binding
  instável + libtorch pesado (~600MB); ONNX exige exportar o modelo e dor
  com ops customizadas.

- **ResNet18 com conv1 adaptada pra 3×3 e maxpool removido**: a conv1 padrão
  da ResNet usa kernel 7×7 com stride 2, que destrói detalhe em imagens 32×32.
  Trocar pra 3×3 stride 1 é o ajuste padrão pra CIFAR.

- **CORS aberto na API**: ok pra demo. Em produção real, restringir ao
  domínio do frontend.

- **`spawn_blocking` no decode da imagem**: decode de PNG/JPEG é CPU-bound;
  rodar inline bloquearia o executor Tokio.

- **`--resume` no treino**: permite continuar de um checkpoint anterior.
  Optimizer/scheduler reiniciam (limitação conhecida), mas os pesos são preservados.

---

## Estrutura

```
.
├── ml/                  # PyTorch
│   ├── scripts/         #   01..05 treino e avaliação
│   ├── sidecar/         #   FastAPI que serve o modelo
│   └── models/          #   cifar10_resnet18.pt (87.76% val acc)
├── api-rust/            # Cargo workspace
│   ├── resize-cli/      #   CLI de redimensionamento (didático)
│   ├── classify-api/    #   API HTTP que faz proxy pro sidecar
│   └── bin/             #   binários compilados (release)
└── frontend/            # React + Vite
```

---

## O que aprendi (e o que ainda quero aprender)

- Por que **CNN > MLP pra imagem** não é "mais profunda" — é viés indutivo de
  localidade + invariância à translação.
- **Transfer learning não é cheating**: o conv1 adaptado pra CIFAR mostra que
  você precisa entender por que a arquitetura original existe pra adaptá-la.
- **Ownership do Rust** força você a pensar em quem é dono do dado. O compilador
  pega bugs que você jamais perceberia em Python.
- A integração **Rust ↔ Python via HTTP** é boring tech no melhor sentido:
  cada lado faz o que faz bem.

Próximos passos:
- [ ] Substituir o sidecar HTTP por gRPC e medir latência
- [ ] Exportar pra ONNX e medir diferença em RAM/cold start
- [ ] Quantização INT8 da ResNet (modelo cai pra ~3MB)
- [ ] Adicionar Grad-CAM no frontend pra mostrar onde a CNN "olhou"
