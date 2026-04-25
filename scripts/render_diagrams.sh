#!/usr/bin/env bash
# Optional: render Mermaid diagrams under experiments/diagrams/ to SVG.
# GitHub renders mermaid in markdown natively, so this is only needed
# for offline / non-GitHub viewers. Requires `mmdc` (Mermaid CLI):
#     npm install -g @mermaid-js/mermaid-cli
# If mmdc is not installed, this script skips silently with a notice.

set -euo pipefail

DIAG_DIR=${DIAG_DIR:-experiments/diagrams}
OUT_DIR=${OUT_DIR:-experiments/RETROSPECTIVE_assets}

if ! command -v mmdc >/dev/null 2>&1; then
    echo "[skip] mmdc not installed — markdown will still render on GitHub."
    echo "       To enable offline SVG: npm install -g @mermaid-js/mermaid-cli"
    exit 0
fi

mkdir -p "${OUT_DIR}"

shopt -s nullglob
for f in "${DIAG_DIR}"/*.mmd; do
    name=$(basename "${f}" .mmd)
    out="${OUT_DIR}/${name}.svg"
    echo "rendering ${f} → ${out}"
    mmdc -i "${f}" -o "${out}" -b transparent
done

echo "rendered $(ls -1 "${OUT_DIR}"/*.svg 2>/dev/null | wc -l) diagrams to ${OUT_DIR}"
