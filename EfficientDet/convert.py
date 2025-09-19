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
import tensorflow as tf # <--- 입력 이름 확인을 위해 추가

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

# +++ 신규 함수: TFLite 모델의 입력 텐서 이름을 가져옵니다. +++
def get_tflite_input_name(tflite_path: Path) -> str:
    """Uses TensorFlow Lite interpreter to get the name of the first input tensor."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    if not input_details:
        raise ValueError("Could not get input details from TFLite model.")
    input_name = input_details[0]['name']
    print(f"[i] Found TFLite input tensor name: {input_name}")
    return input_name

# *** 수정된 함수: NCHW 변환 옵션을 추가합니다. ***
def run_tf2onnx_tflite_to_onnx(tflite_path: Path, onnx_path: Path, tflite_input_name: str, opset: int = 13) -> Path:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", str(tflite_path),
        "--output", str(onnx_path),
        "--opset", str(opset),
        # <--- NCHW 변환을 위한 핵심 옵션 ---
        "--inputs-as-nchw", tflite_input_name,
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
    # NCHW 입력을 위한 shape 명시
    input_shapes = {"image": [1, 3, 320, 320]} # efficientdet은 보통 'image'를 입력 이름으로 가집니다.
    m = onnx.load(str(onnx_path))
    # simplify 실행 시 동적 입력을 고정 shape으로 지정해주면 더 안정적입니다.
    # 모델의 실제 입력 이름을 모를 경우 이 부분은 생략해도 동작할 수 있습니다.
    try:
        input_name = m.graph.input[0].name
        input_shapes = {input_name: [1, 3, 320, 320]}
        print(f"[i] Using input shape for simplifier: {input_shapes}")
        sm, ok = simplify(m, overwrite_input_shapes=input_shapes, check_n=3, perform_optimization=True)
    except Exception:
        print("[w] Failed to simplify with explicit shape, trying without it.")
        sm, ok = simplify(m, check_n=3, perform_optimization=True)

    if not ok:
        print("[!] Simplify check failed; keep original.")
        return onnx_path
    out = onnx_path.with_name(onnx_path.stem + "_sim.onnx")
    onnx.save(sm, str(out))
    print(f"[+] Simplified model saved: {out}")
    return out

def fix_batch_dim_to_1(onnx_path: Path) -> Path:
    if not FIX_BATCH_TO_1:
        return onnx_path

    print("[i] Fixing batch dim (first axis) to 1 ...")
    model = onnx.load(str(onnx_path))

    def set_first_dim_to_1(v):
        tt = v.type.tensor_type
        if tt.HasField("shape") and len(tt.shape.dim) >= 1:
            d0 = tt.shape.dim[0]
            d0.dim_param = ""
            d0.dim_value = 1

    for v in model.graph.input: set_first_dim_to_1(v)
    for v in model.graph.output: set_first_dim_to_1(v)
    for v in model.graph.value_info: set_first_dim_to_1(v)

    tmp_out = onnx_path.with_name(onnx_path.stem + "_bs1.onnx")
    onnx.save(model, str(tmp_out))
    print(f"[+] Batch=1 fixed ONNX saved: {tmp_out}")

    try:
        from onnx import shape_inference
        inf = shape_inference.infer_shapes_path(str(tmp_out))
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
            shp = [d.dim_value if d.dim_value > 0 else d.dim_param or "?" for d in v.type.tensor_type.shape.dim]
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
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        # NCHW 포맷에 맞는 더미 데이터 생성
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

    # *** 1.5) Get TFLite input tensor name ***
    tflite_input_name = get_tflite_input_name(tflite_path)

    # 2) Convert (TFLite -> ONNX with NCHW input)
    # *** 입력 이름을 인자로 전달 ***
    onnx_path = run_tf2onnx_tflite_to_onnx(tflite_path, onnx_path, tflite_input_name, opset=OPSET)

    # 3) Simplify
    onnx_path = simplify_onnx(onnx_path)

    # 4) Fix batch=1
    onnx_path = fix_batch_dim_to_1(onnx_path)

    # 5) Inspect / Quick inference
    inspect_model(onnx_path)
    quick_infer(onnx_path)

def main():
    # 필요한 라이브러리 설치 확인
    try:
        import tf2onnx
        import onnxsim
    except ImportError as e:
        print(f"[!] Missing required library: {e.name}")
        print("Please run: pip install tensorflow tf2onnx onnxsim onnxruntime")
        sys.exit(1)

    for name, url in MODELS.items():
        convert_one(name, url)
    print("\n[✓] All done!")

if __name__ == "__main__":
    main()
