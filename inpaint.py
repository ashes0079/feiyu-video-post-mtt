#!/usr/bin/env python3
"""LaMa 图像修补（MUSA/CUDA 通用，fp32）"""
import argparse, sys, time
import numpy as np
try:
    import torchada  # noqa: 让 torch.cuda 映射到 MUSA（必须最先导入）
except Exception:
    pass
import torch
from PIL import Image

def load(p):
    return np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0

def load_mask(p):
    m = np.array(Image.open(p).convert("L")).astype(np.float32)
    return (m > 0.5).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="/root/autodl-tmp/w/big-lama.pt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torch.jit.load(a.weights, map_location=dev).float().to(dev).eval()
    img, msk = load(a.image), load_mask(a.mask)
    h, w = img.shape[:2]
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    it = torch.from_numpy(np.pad(img, ((0,ph),(0,pw),(0,0)), mode="symmetric")).permute(2,0,1).unsqueeze(0).to(dev).float()
    mt = torch.from_numpy(np.pad(msk, ((0,ph),(0,pw)), mode="symmetric")).unsqueeze(0).unsqueeze(0).to(dev).float()
    with torch.no_grad():
        out = m(it, mt)
    arr = out[0].permute(1,2,0).detach().float().cpu().numpy()[:h,:w]
    Image.fromarray((np.clip(arr,0,1)*255).round().astype(np.uint8)).save(a.out)
    print("saved", a.out)

if __name__ == "__main__":
    main()