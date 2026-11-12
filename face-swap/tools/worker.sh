#!/usr/bin/env bash
# 单个并行 worker：按文件名序号取模分片，跳过已完成的帧
set -uo pipefail
IDX=$1; TOT=$2
WORK=${WORK:-/root/work}; SRC=${SRC:?}
REF_FRAME=${REF_FRAME:-240}; REF_POS=${REF_POS:-0}
FF_DIR=${FF_DIR:-/root/ff}
PY=${PY:-python3}

i=0
for F in "$WORK"/frames_in/*.png; do
  i=$((i+1))
  [ $(( (i-1) % TOT )) -ne "$IDX" ] && continue
  N=$(basename "$F")
  [ -f "$WORK/frames_out/$N" ] && continue
  cd "$FF_DIR"
  "$PY" facefusion.py headless-run -s "$SRC" -t "$F" -o "$WORK/frames_out/$N" \
    --processors face_swapper face_enhancer \
    --execution-providers cuda \
    --face-swapper-model inswapper_128_fp16 \
    --face-swapper-pixel-boost 512x512 \
    --face-enhancer-model gpen_bfr_512 \
    --face-selector-mode reference \
    --reference-frame-number "$REF_FRAME" --reference-face-position "$REF_POS" \
    > /dev/null 2>&1
  [ -f "$WORK/frames_out/$N" ] || cp "$F" "$WORK/frames_out/$N"   # 没脸 → 保留原帧
done