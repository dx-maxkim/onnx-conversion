import torch
import os
import timm
import onnx
from onnxsim import simplify


# ILSVRC 2012, 1,000 클래스
MODELS = [
    "tf_efficientnetv2_s",
    "tf_efficientnetv2_m",
    "tf_efficientnetv2_l",
]

# timm default_cfg 기반 입력 크기
# Ref: https://github.com/NVlabs/A-ViT/blob/master/timm/models/efficientnet.py -> default_cfg
IM_SIZE = {
    "tf_efficientnetv2_s": 300,
    "tf_efficientnetv2_m": 384,
    "tf_efficientnetv2_l": 384,
}

os.makedirs("onnx_out", exist_ok=True)

def export_one(model_name: str, opset: int = 13):
    sz = IM_SIZE[model_name]
    print(f"[+] exporting {model_name} (input {sz}x{sz}) ...")
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    dummy = torch.randn(1, 3, sz, sz)  # batch=1 고정
    onnx_path = os.path.join("onnx_out", f"{model_name.replace('.', '_')}.onnx")
    onnx_simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")

    # PyTorch -> ONNX
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,  # 상수 전개
        # dynamic_axes 미지정 -> 완전 고정 shape로 export
    )

    # 검증
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # onnxsim으로 단순화 (+ 입력형상 고정)
    print(f"[+] simplifying {onnx_path} ...")
    simplified, ok = simplify(
        onnx_model,
        overwrite_input_shapes={"input": [1, 3, sz, sz]}
    )
    assert ok, "onnxsim check failed"
    onnx.save(simplified, onnx_simplified_path)
    print(f"[✓] saved: {onnx_simplified_path}")

if __name__ == "__main__":
    for m in MODELS:
        export_one(m)
    print("[✓] all done")

