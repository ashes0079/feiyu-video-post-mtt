# 视频批量换脸一键包（Linux + NVIDIA GPU）

把一段视频里的**指定一个人**换成另一张脸，逐帧稳定、可并行、跑完自动合成带原声成片。
本包是实跑出来的流水线，不是论文复现：1080p / 25fps / 93 秒 / 2325 帧，单卡 RTX 3090 约 1.5 小时出片。

## 为什么不用 FaceFusion 的视频模式

FaceFusion 的 video 模式在**没有脸的帧上会直接崩**（进程退出、无报错），而一段 MV 里
大约 1/3 的帧是远景、腿部、背影。所以本包的做法是：

1. **抽帧**成图片
2. **逐帧判定**：送进 FaceFusion，成功就留结果，失败（= 这一帧没脸）就**复制原帧**
3. 帧序列**原样合成**视频，音轨从原片直接取

好处：不会因为某一帧没脸就整段失败；坏处：逐帧独立推理没有时序约束，会有轻微"抽动"，
要更稳就走训练式（DeepFaceLab）。

## 环境

```bash
# 1) 装 FaceFusion（不要用它自带的 requirements，会装 CPU 版 onnxruntime）
git clone --depth 1 https://github.com/facefusion/facefusion.git
pip install numpy==2.5.4 opencv-python-headless==4.11.0.86 tqdm scipy onnx pillow requests pyyaml imageio imageio-ffmpeg

# 2) 装 GPU 版 onnxruntime（关键：版本要和 CUDA 匹配）
pip install --no-deps onnxruntime-gpu==1.22.0     # CUDA 12.x 卡
# Blackwell(50系/PRO 6000) 需要 CUDA 13 版，见 docs/params.md

# 3) CUDA 运行库（cuDNN 9 / cuBLAS）走 pip，配 LD_LIBRARY_PATH
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12
export SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# 4) 自检：必须看到 CUDAExecutionProvider，且跑一帧时 nvidia-smi 显存 > 0
python -c "import onnxruntime as o;print(o.get_available_providers())"
```

## 用法

```bash
export SRC=/path/face.jpg          # 源脸（正脸、清晰、无遮挡）
export VIDEO=/path/video.mp4       # 原片
export WORK=/root/work             # 工作目录
export REF_FRAME=240               # 目标人物脸最清楚的那一帧（25fps 下的帧号）
export REF_POS=0                   # 那一帧里目标脸是第几张（从左到右，一般 0）
export JOBS=8                      # 并行进程数（3090 用 8，12G 卡用 4）

bash run_batch.sh
```

产物：`$WORK/out.mp4`（与原片同分辨率同帧率，带原声）。

## 关键坑（都踩过）

| 现象 | 原因 | 处理 |
|---|---|---|
| 逐帧跑整段直接退出、无报错 | 没脸的帧会让 FaceFusion 崩 | 逐帧跑，失败就复制原帧 |
| 一张图跑 50 秒 | 机器上同时有 CPU 版和 GPU 版 onnxruntime，加载了 CPU 版 | 只留 GPU 版，看 `get_available_providers()` |
| 换错人 | 默认模式挑"最大的脸" | 用 `--face-selector-mode reference` 锁定参考帧 |
| 脸边缘闪 | 遮罩逐帧变 | 加 `--face-mask-types`，或关掉 face_enhancer |
| 合成后没声音 | 逐帧合成丢了音轨 | 用原片当第二个输入 `-map 1:a:0` |

## 实测

| 机器 | 每帧 | 93 秒片（2325 帧） |
|---|---|---|
| RTX 3090 24G（8 并行） | ≈3.3 秒 | ≈1.5 小时 |
| RTX 4070Ti 12G（8 并行） | ≈6–8 秒 | ≈3 小时 |
| 摩尔线程 MTT S4000（单进程） | 未跑换脸，仅跑过 LaMa/ESRGAN | — |

## 目录

```
run_batch.sh            一键：抽帧 → 逐帧换 → 合成
tools/worker.sh         单个并行 worker（按模数分片）
tools/filter_faces.py   可选的"有脸预筛"（Haar，仅供省算力，慎用）
docs/params.md          参数对照与版本兼容
```

## 许可与合规

- FaceFusion 为 OpenRAIL-AS：允许商用，**禁止欺骗性用途**
- 用别人的脸必须拿**书面授权**（用途、平台、期限、是否允许二次修改）
- 成片必须标注「本片含 AI 换脸合成效果，已获本人书面授权」
- 本仓库不含任何真实人物素材
