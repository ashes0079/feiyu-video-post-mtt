#!/usr/bin/env python3
"""MTT S4000 基准：矩阵乘 / 卷积（LaMa、超分请用 inpaint.py / upscale.py 跑真实文件）"""
import time
try:
    import torchada  # noqa: 让 torch.cuda 映射到 MUSA
except Exception:
    pass
import torch


def bench(fn, n=5):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n


def main():
    d = "cuda"
    name = torch.musa.get_device_name(0) if getattr(torch, "musa", None) and torch.musa.is_available() else torch.cuda.get_device_name(0)
    print("设备:", name)
    N = 2048
    x = torch.randn(N, N, device=d, dtype=torch.float16)
    y = torch.randn(N, N, device=d, dtype=torch.float16)
    t = bench(lambda: x @ y, 10)
    print(f"matmul {N}^2 fp16: {t*1000:.1f}ms  {2*N**3/t/1e12:.1f} TFLOP/s")
    del x, y
    torch.cuda.empty_cache()
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(d).half()
    im = torch.randn(1, 3, 540, 960, device=d, dtype=torch.float16)
    print(f"conv 540p fp16: {bench(lambda: conv(im), 5)*1000:.1f}ms")
    print("（LaMa / 超分基准请看 README 里的实测值，或用真实文件跑 inpaint.py / upscale.py）")


if __name__ == "__main__":
    main()