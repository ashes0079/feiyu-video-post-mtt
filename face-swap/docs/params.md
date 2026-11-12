# 参数对照与版本兼容

## 换脸模型（`--face-swapper-model`）

| 模型 | 显存 | 质量 | 说明 |
|---|---|---|---|
| `inswapper_128` | ~2G | 中 | 原版，稳 |
| **`inswapper_128_fp16`** | ~2G | 中 | **本包默认**，半精度，速度稍快 |
| `hyperswap_1a_256` | ~3G | 中上 | 细节更好，边缘略硬 |
| `simswap_unofficial_512` | ~3G | 中 | 512 输入，慢 |
| `uniface_256` | ~3G | 中 | 表情保持较好 |
| `ghost_3_256` | ~3G | 中上 | 换完更"像同一个人"，但偏柔 |
| `blendswap_256` | ~2G | 低 | 只做融合，做风格化 |

## 像素增强（`--face-swapper-pixel-boost`）

`256x256`（默认，快）/ `512x512`（**本包默认**，细节明显好）/ `768x768` / `1024x1024`（慢，12G 卡慎用）。
实测同机 512 比 256 慢约 1.6 倍，但下颌线、眉毛清晰度提升明显。

## 人脸修复（`--face-enhancer-model`）

`gfpgan_1.4`（轻）/ **`gpen_bfr_512`（本包默认）** / `codeformer`（保真，慢）/ `gpen_bfr_1024`（重）。
注意：修复是逐帧独立做，会放大闪烁；如果成片"抽"，先关它。

## 选脸（`--face-selector-mode`）

- `one`（默认）：挑画面里最大的一张脸 —— **多人物镜头必换错人**
- `reference`：按参考帧锁定目标脸 —— **本包默认**
  - `--reference-frame-number`：目标脸最清楚的帧号（25fps 时 = 秒数 × 25）
  - `--reference-face-position`：那一帧里目标脸排第几（0 起，从左到右）

## onnxruntime 与 CUDA 版本

| 显卡 | onnxruntime-gpu | CUDA |
|---|---|---|
| RTX 20/30/40 系 | 1.22.0 | 12.x + cuDNN 9 |
| Blackwell（50 系 / PRO 6000） | 需 CUDA 13 构建 | 13.x |

**必看**：`python -c "import onnxruntime as o; print(o.get_available_providers())"`
只有 `CPUExecutionProvider` = 白跑（一张图 50 秒），要检查：
1. 是否同时装了 CPU 版 `onnxruntime`（卸载它）
2. cuDNN/cuBLAS 是否在 `LD_LIBRARY_PATH`
3. 跑的时候 `nvidia-smi` 显存是否 > 0

## 性能实测

| 配置 | 每帧（1080p，512 增强 + gpen512） |
|---|---|
| RTX 3090 + 8 并行 | ≈3.3 秒（总体吞吐） |
| RTX 4070Ti 12G + 8 并行 | ≈6–8 秒 |
| 单进程（含进程启动开销） | ≈10–20 秒 |

> 逐帧跑的主要开销是**每帧重新加载模型**（约 4–10 秒），所以并行把固定开销摊掉是必须的。
