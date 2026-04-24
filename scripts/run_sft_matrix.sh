#!/usr/bin/env bash
# Run the full exp/14 SFT matrix: pre-SFT eval + SFT + post-SFT eval for
# every pretrained checkpoint. Idempotent — a stage is skipped if its
# output already exists, so interrupted runs can be resumed by re-running.
#
# Budget (single 5090): ~1 h pre-SFT eval total + ~4 h SFT × 8 + ~1 h
# post-SFT eval × 8 ≈ ~42 h wall-clock.
#
# Usage:
#   bash scripts/run_sft_matrix.sh                    # full matrix
#   CHECKPOINTS="03-modded-tricks" bash scripts/run_sft_matrix.sh   # one
#   SFT_TOKENS=1000000 bash scripts/run_sft_matrix.sh  # short smoke

set -euo pipefail

CHECKPOINTS=${CHECKPOINTS:-"baseline 01-modern-block 02-muon 03-modded-tricks 05-speed-pack 06-muon-mup 10-mla 11-loopllm"}
SFT_TOKENS=${SFT_TOKENS:-500000000}
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "${LOG_DIR}"

for CKPT in ${CHECKPOINTS}; do
    PRETRAIN_CKPT="runs/${CKPT}/best_val.pt"
    SFT_DIR="runs/sft-${CKPT}"
    PRESFT_EVAL="runs/${CKPT}/eval_results.json"
    POSTSFT_EVAL="${SFT_DIR}/eval_results.json"

    if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
        echo "[skip] ${CKPT}: no checkpoint at ${PRETRAIN_CKPT}"
        continue
    fi

    # ---- Pre-SFT baseline eval -------------------------------------------
    if [[ -f "${PRESFT_EVAL}" ]]; then
        echo "[skip] ${CKPT} pre-SFT eval: ${PRESFT_EVAL} exists"
    else
        echo "==> ${CKPT}: pre-SFT eval"
        PYTHONUNBUFFERED=1 uv run python scripts/sft_eval.py \
            --run-dir "runs/${CKPT}" \
            2>&1 | tee "${LOG_DIR}/sft-eval-pre-${CKPT}.log"
    fi

    # ---- SFT -------------------------------------------------------------
    if [[ -f "${SFT_DIR}/best_val.pt" ]]; then
        echo "[skip] ${CKPT} SFT: ${SFT_DIR}/best_val.pt exists"
    else
        echo "==> ${CKPT}: SFT (${SFT_TOKENS} tokens)"
        PYTHONUNBUFFERED=1 uv run python scripts/sft.py \
            --pretrain-ckpt "${PRETRAIN_CKPT}" \
            --run-dir "${SFT_DIR}" \
            --sft-tokens "${SFT_TOKENS}" \
            2>&1 | tee "${LOG_DIR}/sft-${CKPT}.log"
    fi

    # ---- Post-SFT eval ---------------------------------------------------
    if [[ -f "${POSTSFT_EVAL}" ]]; then
        echo "[skip] ${CKPT} post-SFT eval: ${POSTSFT_EVAL} exists"
    else
        echo "==> ${CKPT}: post-SFT eval"
        PYTHONUNBUFFERED=1 uv run python scripts/sft_eval.py \
            --run-dir "${SFT_DIR}" \
            2>&1 | tee "${LOG_DIR}/sft-eval-post-${CKPT}.log"
    fi

    echo "==> ${CKPT} done."
done

echo "matrix complete. running aggregator..."
PYTHONUNBUFFERED=1 uv run python scripts/sft_matrix_report.py \
    --checkpoints ${CHECKPOINTS} \
    --out experiments/14-sft-matrix/matrix.json
