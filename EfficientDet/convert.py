#!/usr/bin/env python3
import os
import sys
import json
import subprocess
from pathlib import Path

import requests
import onnx
import numpy as np
import onnxruntime as ort

# =========================
# User-config (하드코딩)
# =========================
OPSET = 13
DOWNLOAD_DIR = Path("models")
ONNX_DIR = Path("onnx_out")
RUN_SIMPLIFY = True
RUN_INFER_CHECK = True
FIX_BATCH_TO_1 = True

# Mediapipe EfficientDet-Lite TFLite URLs
# https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector
MODELS = {
    "efficientdet_lite0": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite",
    "efficientdet_lite2": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite",
}

# =========================
# Utils
# =========================
def download_tflite(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[i] Found existing TFLite: {out_path}")
        return out_path
    print(f"[i] Downloading TFLite from {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[+] Saved: {out_path}")
    return out_path

def run_tf2onnx_tflite_to_onnx(tflite_path: Path, onnx_path: Path, opset: int = 13) -> Path:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", str(tflite_path),
        "--output", str(onnx_path),
        "--opset", str(opset),
    ]
    print("[i] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[+] ONNX saved: {onnx_path}")
    return onnx_path

def simplify_onnx(onnx_path: Path) -> Path:
    if not RUN_SIMPLIFY:
        return onnx_path
    try:
        from onnxsim import simplify
    except Exception as e:
        print(f"[i] onnx-simplifier not available ({e}); skipping simplify.")
        return onnx_path

    print("[i] Simplifying ONNX with onnx-simplifier...")
    m = onnx.load(str(onnx_path))
    sm, ok = simplify(m, check_n=3, perform_optimization=True)
    if not ok:
        print("[!] Simplify check failed; keep original.")
        return onnx_path
    out = onnx_path.with_name(onnx_path.stem + "_sim.onnx")
    onnx.save(sm, str(out))
    print(f"[+] Simplified model saved: {out}")
    return out

def fix_batch_dim_to_1(onnx_path: Path) -> Path:
    """
    모든 입력/출력 텐서의 첫 번째 축을 1로 고정합니다(랭크>=1인 경우).
    모델 전반 shape-consistency를 위해 shape inference를 한 번 더 수행합니다.
    """
    if not FIX_BATCH_TO_1:
        return onnx_path

    print("[i] Fixing batch dim (first axis) to 1 ...")
    model = onnx.load(str(onnx_path))

    def set_first_dim_to_1(v):
        tt = v.type.tensor_type
        if tt.HasField("shape") and len(tt.shape.dim) >= 1:
            d0 = tt.shape.dim[0]
            # dim_param or dim_value 어떤 것이든 1로 고정
            d0.dim_param = ""
            d0.dim_value = 1

    # graph inputs/outputs
    for v in model.graph.input:
        set_first_dim_to_1(v)
    for v in model.graph.output:
        set_first_dim_to_1(v)

    # value_info (중간 텐서 정보가 있다면 같이 고정)
    for v in model.graph.value_info:
        set_first_dim_to_1(v)

    tmp_out = onnx_path.with_name(onnx_path.stem + "_bs1.onnx")
    onnx.save(model, str(tmp_out))
    print(f"[+] Batch=1 fixed ONNX saved: {tmp_out}")

    # shape inference (선택적이지만 권장)
    try:
        from onnx import shape_inference
        inf = shape_inference.infer_shapes_path(str(tmp_out))
        # infer_shapes_path가 파일을 덮어쓰는 구현이 아닌 경우 대비
        if isinstance(inf, onnx.ModelProto):
            onnx.save(inf, str(tmp_out))
    except Exception as e:
        print(f"[i] shape_inference skipped ({e})")

    return tmp_out

def inspect_model(onnx_path: Path):
    m = onnx.load(str(onnx_path))
    g = m.graph
    def io_list(vs):
        out = []
        for v in vs:
            shp = []
            if v.type.tensor_type.shape.dim:
                for d in v.type.tensor_type.shape.dim:
                    if d.dim_value != 0:
                        shp.append(d.dim_value)
                    elif d.dim_param:
                        shp.append(d.dim_param)
                    else:
                        shp.append("?")
            out.append({"name": v.name, "shape": shp})
        return out
    info = {"inputs": io_list(g.input), "outputs": io_list(g.output), "nodes": len(g.node)}
    print("[i] Model IO:")
    print(json.dumps(info, indent=2))
    return info

def quick_infer(onnx_path: Path):
    if not RUN_INFER_CHECK:
        return
    print("[i] Quick inference...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {}
    for i in sess.get_inputs():
        shape = []
        for d in i.shape:
            if isinstance(d, int) and d > 0:
                shape.append(d)
            else:
                # 동적/미지정 -> 1로 대체
                shape.append(1)
        # 일반적으로 NHWC(1,H,W,3) 또는 NCHW(1,3,H,W)일 수 있음
        arr = np.random.rand(*shape).astype(np.float32)
        feeds[i.name] = arr
    outs = sess.run(None, feeds)
    for idx, o in enumerate(outs):
        print(f"  - out[{idx}] shape={o.shape} dtype={o.dtype}")

# =========================
# Pipeline
# =========================
def convert_one(name: str, url: str):
    print("\n" + "=" * 80)
    print(f"[+] Converting {name}")
    print("=" * 80)

    tflite_path = DOWNLOAD_DIR / f"{name}.tflite"
    onnx_path = ONNX_DIR / f"{name}.onnx"

    # 1) Download
    download_tflite(url, tflite_path)

    # 2) Convert (TFLite -> ONNX)
    onnx_path = run_tf2onnx_tflite_to_onnx(tflite_path, onnx_path, opset=OPSET)

    # 3) Simplify
    onnx_path = simplify_onnx(onnx_path)

    # 4) Fix batch=1
    onnx_path = fix_batch_dim_to_1(onnx_path)

    # 5) Inspect / Quick inference
    inspect_model(onnx_path)
    quick_infer(onnx_path)

def main():
    for name, url in MODELS.items():
        convert_one(name, url)
    print("\n[✓] All done!")

if __name__ == "__main__":
    main()

