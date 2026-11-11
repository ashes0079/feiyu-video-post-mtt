# 摩尔线程（MTT S4000）视频后处理 · 一键部署包

实测环境：AutoDL 摩尔线程专区 MTT S4000 48G · torch 2.11.0.post1 · MUSA 5.2.0 · Python 3.10
实测日期：2026-11-11 · 总花费 **0.96 元**（1.86 元/时）

## 三行上手

```bash
bash setup_mtt.sh /root/autodl-tmp/w        # 装环境 + 下权重（约 10 分钟）
python inpaint.py --image f.png --mask m.png --out out.png   # 去物体/去水印
python upscale.py --image f.png --out up.png                 # 4x 超分
bash run_chain.sh 帧目录 掩膜目录 输出目录                    # 整链（去绳+超分）
```

## 实测能力

| 能力 | 结果 |
|---|---|
| fp16 / bf16 | ✅ 都支持 |
| matmul 2048² fp16 | **48.3 TFLOP/s**（4096² 时 76 TFLOP/s） |
| 540p 卷积 | 0.6 ms/层 |
| **LaMa 修补** | ✅ **1.53 秒/帧**（1024×576，fp32） |
| **Real-ESRGAN 4x** | ✅ **1.17 秒/帧**（720p→4x，fp16） |
| 抖音级 19.3 秒整链（580 帧） | 约 **26 分钟 ≈ 0.8 元** |
| ComfyUI / SVD | ❌ 缺 torchvision-MUSA 算子，跑不了 |
| SAM2 | ❌ GitHub 拉不下来 + 需 CUDA 扩展 |

## 三个必踩的坑（都已在脚本里处理）

1. **pip 会把 torch 升级掉**：装 `diffusers/transformers` 时 pip 自动把 torch 换成 2.14.1（CUDA 版），MUSA 后端直接报废。
   → 脚本里对这些包用 `--no-deps`，并在最后**强制回滚** `/home/pytorch/dist/torch-2.11.0.post1*.whl` + `torch_musa` 同版本。
2. **必须 `import torchada`**：它把 `torch.cuda` 映射到 MUSA（注意 `torch.cuda.is_available()` 仍报 False，但张量能正常落在 `musa:0`，别被这个返回值骗了）。
3. **MUSA 不支持 fp64**：`PadImpl not implemented for 'Double'`。LaMa 的 TorchScript 权重必须 `.float()`，输入 float32。

## 图像/视频编解码

镜像里没有系统 ffmpeg，用 `imageio-ffmpeg` 自带的二进制；视频先拆帧再跑本包，最后合帧。

## 上架分成

AutoDL 创作激励：NVIDIA 镜像 0.06 元/时，**华为 NPU / 摩尔线程镜像 0.21 元/时（3.5 倍）**——本包就是冲着这条来的。