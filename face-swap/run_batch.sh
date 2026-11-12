#!/usr/bin/env bash
# 一键批量换脸：抽帧 → 逐帧换脸（失败保留原帧）→ 合成带原声成片
set -euo pipefail

SRC=${SRC:?需要 SRC=源脸图片}
VIDEO=${VIDEO:?需要 VIDEO=原片路径}
WORK=${WORK:-/root/work}
REF_FRAME=${REF_FRAME:-240}
REF_POS=${REF_POS:-0}
JOBS=${JOBS:-8}
FF_DIR=${FF_DIR:-/root/ff}          # FaceFusion 目录
FPS=${FPS:-25}
SCALE=${SCALE:-1920}               # 处理分辨率（宽）

mkdir -p "$WORK"/{frames_in,frames_out}
export WORK SRC REF_FRAME REF_POS FPS

echo "[1/4] 抽帧 $FPS fps, 宽 $SCALE"
ffmpeg -y -loglevel error -i "$VIDEO" -vf "scale=$SCALE:-2,fps=$FPS" -vsync 0 "$WORK/frames_in/%04d.png"
N=$(ls "$WORK/frames_in" | wc -l)
echo "     共 $N 帧"

echo "[2/4] 逐帧换脸，$JOBS 并行"
for k in $(seq 0 $((JOBS-1))); do
  nohup bash "$(dirname "$0")/tools/worker.sh" "$k" "$JOBS" > "$WORK/w$k.log" 2>&1 &
done
wait                               # 等全部 worker 结束

echo "[3/4] 缺口回填（没脸 / 失败的帧用原帧）"
i=0
for f in "$WORK"/frames_in/*.png; do
  i=$((i+1)); n=$(basename "$f")
  [ -f "$WORK/frames_out/$n" ] || cp "$f" "$WORK/frames_out/$n"
done
echo "     $i 帧齐全"

echo "[4/4] 合成（保留原声）"
ffmpeg -y -loglevel error -framerate "$FPS" -i "$WORK/frames_out/%04d.png" -i "$VIDEO" \
  -map 0:v:0 -map 1:a:0 -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -c:a aac -shortest \
  "$WORK/out.mp4"

echo "完成: $WORK/out.mp4"