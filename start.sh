#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── sidecar Python ────────────────────────────────────────────────────────────
echo "[1/3] Iniciando sidecar Python (porta 8001)..."
cd "$ROOT/ml"
.venv/bin/uvicorn sidecar.main:app --host 127.0.0.1 --port 8001 &
SIDECAR_PID=$!

# aguarda o sidecar carregar o modelo
echo "      Aguardando modelo carregar..."
for i in $(seq 1 15); do
  if curl -sf http://127.0.0.1:8001/health > /dev/null 2>&1; then
    echo "      Sidecar OK"
    break
  fi
  sleep 1
done

# ── API Rust ──────────────────────────────────────────────────────────────────
echo "[2/3] Iniciando API Rust (porta 8000)..."
cd "$ROOT/api-rust"
SIDECAR_URL=http://127.0.0.1:8001 ./target/release/classify-api &
RUST_PID=$!
sleep 1

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "[3/3] Iniciando frontend (porta 5173)..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Tudo rodando:"
echo "  Frontend  → http://127.0.0.1:5173"
echo "  Rust API  → http://127.0.0.1:8000"
echo "  Sidecar   → http://127.0.0.1:8001"
echo ""
echo "Pressione Ctrl+C para encerrar tudo."

# encerra os 3 processos ao sair
trap "kill $SIDECAR_PID $RUST_PID $FRONTEND_PID 2>/dev/null; echo 'Encerrado.'" EXIT INT TERM
wait
