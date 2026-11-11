#!/usr/bin/env bash
# 摩尔线程 MTT S4000 视频后处理环境（一键）
# 用法：bash setup_mtt.sh [权重目录，默认 /root/autodl-tmp/w]
set -u
W="${1:-/root/autodl-tmp/w}"
mkdir -p "$W"
echo "== 1/6 依赖 =="
pip install -q --no-deps opencv-python-headless==4.10.0.84 || true
pip install -q numpy pillow scipy scikit-image imageio imageio-ffmpeg timm || true
# 注意：diffusers/transformers 用 --no-deps 装，否则 pip 会把 torch 升级成 CUDA 版，MUSA 后端直接报废
pip install -q --no-deps diffusers transformers accelerate safetensors huggingface_hub tokenizers regex requests filelock pyyaml tqdm packaging || true
echo "== 2/6 LaMa（TorchScript）=="
pip install -q --no-deps simple-lama-inpainting || true
python - <<PY
import os, urllib.request
p = "$W/big-lama.pt"
if not os.path.exists(p):
    urllib.request.urlretrieve("https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt", p)
print("big-lama:", round(os.path.getsize(p)/1e6,1), "MB")
PY
echo "== 3/6 Real-ESRGAN 权重 =="
python - <<PY
import os, urllib.request
p = "$W/RealESRGAN_x4plus.pth"
if not os.path.exists(p):
    urllib.request.urlretrieve("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", p)
print("RealESRGAN:", round(os.path.getsize(p)/1e6,1), "MB")
PY
echo "== 4/6 Demucs 人声分离 =="
pip install -q demucs librosa soundfile || true
echo "== 5/6 自检 =="
python - <<PY
import torch, cv2, numpy
print("torch", torch.__version__, "| musa", torch.musa.is_available() if hasattr(torch,"musa") else "n/a",
      "| dev", torch.cuda.get_device_name(0))
PY
echo "== 6/6 基准 =="
python "$(dirname "$0")/bench_mtt.py" --weights "$W"
echo "环境就绪。注意：MUSA 不支持 fp64，模型一律用 fp32。"

echo "== 7/7 锁定 torch 版本（MUSA 只认 2.11.0.post1）=="
python - <<'PY'
import subprocess, sys
try:
    import torch, torch.version
    ver = torch.__version__
except Exception:
    ver = ""
if not ver.startswith("2.11.0.post1"):
    print("当前 torch:", ver, "→ 回滚到镜像自带 2.11.0.post1")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                    "/home/pytorch/dist/torch-2.11.0.post1-cp310-cp310-linux_x86_64.whl"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
                    "/home/torch_musa/dist/torch_musa-2.11.0.post1-cp310-cp310-linux_x86_64.whl"], check=False)
else:
    print("torch 已是 MUSA 版")
PY