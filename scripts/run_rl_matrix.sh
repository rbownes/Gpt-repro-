#!/usr/bin/env bash
# Run the full exp/15 RL matrix: pre-RL gen-eval (cached from Phase A)
# + RL + post-RL gen-eval for every SFT'd checkpoint. Idempotent —
# stages skip themselves when their output exists, so interrupted runs
# resume on re-run.
#
# Per-ckpt budget: ~5 min pre-RL eval (cached) + ~10–60 min RL
# (depending on saturation) + ~5 min post-RL eval. Matrix total
# ~5–10 GPU-hr.
#
# Usage:
#   bash scripts/run_rl_matrix.sh                          # full matrix
#   CHECKPOINTS="02-muon" bash scripts/run_rl_matrix.sh    # one
#   N_STEPS=200 bash scripts/run_rl_matrix.sh              # short

set -euo pipefail

CHECKPOINTS=${CHECKPOINTS:-"baseline 01-modern-block 02-muon 03-modded-tricks 05-speed-pack 06-muon-mup 10-mla 11-loopllm"}
N_STEPS=${N_STEPS:-500}
PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-8}
GROUP_SIZE=${GROUP_SIZE:-16}
KL_COEF=${KL_COEF:-0.04}
PEAK_LR=${PEAK_LR:-1e-6}
SAT_THRESH=${SAT_THRESH:-0.95}
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "${LOG_DIR}"

for CKPT in ${CHECKPOINTS}; do
    SFT_DIR="runs/sft-${CKPT}"
    RL_DIR="runs/rl-${CKPT}"
    SFT_CKPT="${SFT_DIR}/best_val.pt"
    PRESFT_GEN="${SFT_DIR}/gen_eval_results.json"
    POSTRL_GEN="${RL_DIR}/gen_eval_results.json"

    if [[ ! -f "${SFT_CKPT}" ]]; then
        echo "[skip] ${CKPT}: no SFT checkpoint at ${SFT_CKPT}"
        continue
    fi

    # ---- Pre-RL gen-eval (Phase A artefact; should already exist) -----
    if [[ -f "${PRESFT_GEN}" ]]; then
        echo "[skip] ${CKPT} pre-RL gen-eval: ${PRESFT_GEN} exists"
    else
        echo "==> ${CKPT}: pre-RL gen-eval"
        PYTHONUNBUFFERED=1 uv run python scripts/gen_eval.py \
            --run-dir "${SFT_DIR}" \
            2>&1 | tee "${LOG_DIR}/rl-pre-${CKPT}.log"
    fi

    # ---- RL training ---------------------------------------------------
    if [[ -f "${RL_DIR}/best_val.pt" ]]; then
        echo "[skip] ${CKPT} RL: ${RL_DIR}/best_val.pt exists"
    else
        echo "==> ${CKPT}: RL (${N_STEPS} steps, P=${PROMPTS_PER_STEP}, G=${GROUP_SIZE}, lr=${PEAK_LR})"
        PYTHONUNBUFFERED=1 uv run python scripts/rl.py \
            --pretrain-ckpt "${SFT_CKPT}" \
            --run-dir "${RL_DIR}" \
            --n-steps "${N_STEPS}" \
            --prompts-per-step "${PROMPTS_PER_STEP}" \
            --group-size "${GROUP_SIZE}" \
            --kl-coef "${KL_COEF}" \
            --peak-lr "${PEAK_LR}" \
            --saturation-threshold "${SAT_THRESH}" \
            2>&1 | tee "${LOG_DIR}/rl-${CKPT}.log"
    fi

    # ---- Post-RL gen-eval ---------------------------------------------
    if [[ -f "${POSTRL_GEN}" ]]; then
        echo "[skip] ${CKPT} post-RL gen-eval: ${POSTRL_GEN} exists"
    else
        echo "==> ${CKPT}: post-RL gen-eval"
        PYTHONUNBUFFERED=1 uv run python scripts/gen_eval.py \
            --run-dir "${RL_DIR}" \
            2>&1 | tee "${LOG_DIR}/rl-post-${CKPT}.log"
    fi

    echo "==> ${CKPT} done."
done

echo "RL matrix complete."
