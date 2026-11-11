#!/usr/bin/env bash
# 整链：帧序列 → LaMa 去绳 → 4x 超分 → 输出帧（后续用 imageio-ffmpeg 合成）
# 用法: bash run_chain.sh 输入帧目录 掩膜目录 输出目录 [/root/autodl-tmp/w]
set -u
IN="${1:?输入帧目录}"; MK="${2:?掩膜目录}"; OUT="${3:?输出目录}"; W="${4:-/root/autodl-tmp/w}"
mkdir -p "$OUT"
i=0
for f in "$IN"/*.jpg "$IN"/*.png; do
  [ -e "$f" ] || continue
  b=$(basename "$f"); stem="${b%.*}"
  m1="$MK/${stem}_mask.png"; [ -e "$m1" ] || m1="$MK/${stem}.png"
  if [ -e "$m1" ]; then
    python inpaint.py --image "$f" --mask "$m1" --out "$OUT/_tmp_$stem.png" --weights "$W/big-lama.pt" || cp "$f" "$OUT/_tmp_$stem.png"
  else
    cp "$f" "$OUT/_tmp_$stem.png"
  fi
  python upscale.py --image "$OUT/_tmp_$stem.png" --out "$OUT/$stem.png" --weights "$W/RealESRGAN_x4plus.pth" || cp "$OUT/_tmp_$stem.png" "$OUT/$stem.png"
  rm -f "$OUT/_tmp_$stem.png"
  i=$((i+1))
  [ $((i % 10)) -eq 0 ] && echo "  已处理 $i 帧"
done
echo "完成 $i 帧 -> $OUT"