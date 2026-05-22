# Projeto Classificar Imagem

## Visão Geral

Aplicação fullstack de classificação de imagens ponta a ponta, construída com uma arquitetura de três camadas: frontend em React, gateway em Rust e sidecar de inferência em Python com PyTorch.

O usuário faz upload de uma imagem pelo browser, e o sistema retorna a classe predita com nível de confiança e um ranking das top-K classes mais prováveis, renderizado em barras de progresso.

O modelo é uma ResNet18 com fine-tuning no dataset CIFAR-10, atingindo 87.76% de acurácia de validação.

| Camada | Tecnologia |
|---|---|
| Modelo | PyTorch, ResNet18 fine-tuned no CIFAR-10 |
| API | Rust + Axum (gateway + pré-processamento) |
| Sidecar | Python + FastAPI (inferência) |
| Frontend | React + Vite (upload + visualização do top-K) |

---

## Objetivo

O projeto foi desenvolvido como aplicação prática de conceitos de machine learning e engenharia de software, com foco em:

- Aplicar transfer learning com uma arquitetura CNN estabelecida (ResNet18)
- Construir um pipeline de inferência desacoplado (gateway Rust + sidecar Python)
- Integrar pré-processamento de imagem em baixo nível (redimensionamento Lanczos3 no gateway)
- Expor o resultado de forma visual e interativa no frontend

---

## Funcionalidades

- Upload de imagem direto pelo browser (arrastar ou selecionar arquivo)
- Predição automática da classe com percentual de confiança
- Ranking das top-K classes mais prováveis com barras de progresso
- Latência local típica de 50–80ms do upload até a resposta

---

## Classes suportadas

O modelo classifica imagens nas 10 categorias do CIFAR-10:

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

> O modelo foi treinado em imagens 32×32 do CIFAR-10. Funciona melhor com imagens simples e centralizadas nessas categorias — fotos complexas ou fora dessas classes retornarão um resultado, mas com baixa confiança.

---

## Resultado

| Modelo | Dataset | Params | Val acc |
|---|---|---:|---:|
| ResNet18 transfer learning | CIFAR-10 | 11M | **87.76%** |

Estado da arte no CIFAR-10 é ~99%; 87.76% é uma baseline sólida com transfer learning simples sem ajuste de hiperparâmetros.

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

> **Já tem um modelo treinado?** Coloque o arquivo `.pt` em `ml/models/cifar10_resnet18.pt` antes de rodar o setup — o treino será pulado automaticamente.

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

