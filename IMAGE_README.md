# feiyu-video-post-mtt · 国产卡视频后处理（摩尔线程 MUSA）

基于 **PyTorch 2.11.0 + MUSA 5.2.0** 的国产卡视频后处理镜像：一行装环境，三行跑活。

## 镜像里有什么

- LaMa 大遮罩修补（TorchScript，已按 MUSA 改 fp32）
- Real-ESRGAN 4x 超分（RRDBNet，不依赖 basicsr）
- Demucs 人声分离
- 一键脚本：`setup_mtt.sh` / `inpaint.py` / `upscale.py` / `run_chain.sh` / `bench_mtt.py`
- 代码仓库：https://github.com/ashes0079/feiyu-video-post-mtt

## 实测（MTT S4000 48G，2026-11-11）

| 项目 | 结果 |
|---|---|
| matmul 4096² fp16 | 76 TFLOP/s |
| LaMa 修补 1024×576 | **1.53 秒/帧** |
| Real-ESRGAN 4x（720p） | **1.17 秒/帧** |
| 19.3 秒视频整链（580 帧） | 约 26 分钟 / 0.8 元 |

## 快速开始

```bash
cd /root/autodl-tmp/mtt_pkg
bash setup_mtt.sh /root/autodl-tmp/w          # 装依赖 + 下权重
python inpaint.py --image f.png --mask m.png --out out.png
python upscale.py --image f.png --out up.png
bash run_chain.sh 帧目录 掩膜目录 输出目录      # 整链
```

## 适用 / 不适用

**适用**：去水印、去字幕、去物体、旧素材超分、批量后处理。

**不适用**：ComfyUI、SVD 图生视频、SAM2 —— 它们依赖 `torchvision` 的 C++ 算子，摩尔线程暂无对应轮子（已实测确认）。

## 三个坑（镜像里已处理）

1. **pip 会悄悄升级 torch**（装 diffusers/transformers 时）→ 必须 `--no-deps`，并把 torch 锁回 2.11.0.post1
2. **必须 `import torchada`** → 否则 `torch.cuda` 映射不到 MUSA（`is_available()` 返回 False 属正常）
3. **不支持 fp64** → 模型一律 fp32

## 许可

脚本部分 MIT；模型权重（LaMa / Real-ESRGAN / Demucs）遵循各自上游许可，使用时请遵守。
