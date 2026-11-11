#!/usr/bin/env python3
"""Real-ESRGAN 4x 超分（MUSA/CUDA 通用，fp16）"""
import argparse, sys
import numpy as np
try:
    import torchada  # noqa: 让 torch.cuda 映射到 MUSA（必须最先导入）
except Exception:
    pass
import torch
from PIL import Image

sys.path.insert(0, __file__.rsplit("/",1)[0])
from rrdbnet import RRDBNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="/root/autodl-tmp/w/RealESRGAN_x4plus.pth")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    sd = torch.load(a.weights, map_location="cpu")
    model.load_state_dict(sd.get("params_ema", sd.get("params")), strict=True)
    model = model.to(dev).half().eval()
    img = np.array(Image.open(a.image).convert("RGB")).astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(dev).half()
    with torch.no_grad():
        y = model(x)
    arr = (y.clamp(0,1).float().cpu().squeeze(0).permute(1,2,0).numpy()*255).round().astype(np.uint8)
    Image.fromarray(arr).save(a.out)
    print("saved", a.out, arr.shape)

if __name__ == "__main__":
    main()