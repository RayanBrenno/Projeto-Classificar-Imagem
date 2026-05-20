#!/usr/bin/env bash
# Primeira execução: cria o ambiente Python, treina o modelo e compila a API Rust.
# Após rodar este script, use ./start.sh para subir tudo.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "================================================"
echo "  Setup — classify-app"
echo "================================================"

# ── 1. Ambiente Python ────────────────────────────────────────────────────────
echo ""
echo "[1/5] Criando ambiente virtual Python..."
cd "$ROOT/ml"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "      venv criado."
else
    echo "      venv já existe, pulando."
fi

echo "[2/5] Instalando dependências Python..."
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements.txt --quiet
echo "      Dependências instaladas."

# ── 2. Modelo ─────────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Verificando modelo treinado..."

MODEL="$ROOT/ml/models/cifar10_resnet18.pt"

if [ -f "$MODEL" ]; then
    echo "      Modelo já existe em ml/models/cifar10_resnet18.pt, pulando treino."
else
    echo "      Modelo não encontrado. Iniciando treino (pode demorar bastante sem GPU)..."
    echo "      Dica: acompanhe o progresso abaixo. Ctrl+C cancela o treino."
    echo ""
    cd "$ROOT/ml"
    .venv/bin/python scripts/04_cifar10_resnet.py --epochs 15
    echo ""
    echo "      Treino concluído. Modelo salvo em ml/models/cifar10_resnet18.pt"
fi

# ── 3. Dependências Node ──────────────────────────────────────────────────────
echo ""
echo "[4/5] Instalando dependências do frontend..."
cd "$ROOT/frontend"

if [ ! -d "node_modules" ]; then
    npm install --silent
    echo "      node_modules instalado."
else
    echo "      node_modules já existe, pulando."
fi

# ── 4. Compilar Rust ──────────────────────────────────────────────────────────
echo ""
echo "[5/5] Compilando API Rust (release)..."
cd "$ROOT/api-rust"

if [ -f "target/release/classify-api" ]; then
    echo "      Binário já existe, pulando compilação."
else
    echo "      Compilando... (pode levar alguns minutos na primeira vez)"
    cargo build --release --quiet
    echo "      Compilação concluída."
fi

# ── Pronto ────────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  Setup concluído!"
echo "  Execute ./start.sh para subir a aplicação."
echo "================================================"
