#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""filter_faces.py — 用 Haar 级联筛选含人脸的帧并复制到输出目录。

用法:
    python filter_faces.py 输入帧目录 输出帧目录 [--scale 0.5]
"""

import argparse
import os
import shutil
import sys

import cv2

VALID_EXTS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="检测目录中含人脸的图片并复制到输出目录。"
    )
    parser.add_argument("input_dir", help="输入帧目录")
    parser.add_argument("output_dir", help="输出帧目录")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="检测前图像缩放比例（0 < scale <= 1），默认 0.5",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"错误：输入目录不存在: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    if not (0 < args.scale <= 1):
        print("错误：--scale 必须在 (0, 1] 范围内", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"错误：无法加载 Haar 级联: {cascade_path}", file=sys.stderr)
        sys.exit(1)

    # 收集 png/jpg 文件并按文件名排序
    files = sorted(
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
        and os.path.isfile(os.path.join(args.input_dir, f))
    )

    total = len(files)
    with_face = 0
    without_face = 0
    skipped = 0
    no_face_files = []

    for name in files:
        src_path = os.path.join(args.input_dir, name)
        try:
            img = cv2.imread(src_path)
            if img is None:
                raise ValueError("cv2.imread 读取失败")

            # 先缩小再检测，提速
            if args.scale != 1.0:
                img = cv2.resize(
                    img,
                    None,
                    fx=args.scale,
                    fy=args.scale,
                    interpolation=cv2.INTER_AREA,
                )

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                with_face += 1
                shutil.copy2(src_path, os.path.join(args.output_dir, name))
            else:
                without_face += 1
                no_face_files.append(name)
        except Exception as exc:
            skipped += 1
            print(f"警告：处理 {name} 失败（{exc}），已跳过", file=sys.stderr)

    print(f"总帧数: {total}")
    print(f"有人脸帧数: {with_face}")
    print(f"无人脸帧数: {without_face}")
    print("无人脸帧: " + ",".join(no_face_files))
    if skipped:
        print(f"跳过数量: {skipped}")


if __name__ == "__main__":
    main()