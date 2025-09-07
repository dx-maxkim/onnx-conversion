import onnx
from onnx import helper, TensorProto

def nhwc_to_nchw(onnx_path: str, out_path: str):
    model = onnx.load(onnx_path)
    graph = model.graph

    # 원래 입력
    input_tensor = graph.input[0]
    old_name = input_tensor.name
    new_name = old_name + "_nhwc"

    # 입력 이름 교체
    input_tensor.name = new_name
    input_tensor.type.tensor_type.shape.dim[1].dim_value = 320  # height
    input_tensor.type.tensor_type.shape.dim[2].dim_value = 320  # width
    input_tensor.type.tensor_type.shape.dim[3].dim_value = 3    # channel

    # 새로운 NCHW 입력 정의
    nchw_input = helper.make_tensor_value_info(
        old_name, TensorProto.FLOAT, [1, 3, 320, 320]
    )
    graph.input.insert(0, nchw_input)

    # Transpose 노드 추가 (N,H,W,C → N,C,H,W)
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[old_name],
        outputs=[new_name],
        perm=[0, 3, 1, 2]  # NHWC → NCHW
    )
    graph.node.insert(0, transpose_node)

    # 저장
    onnx.save(model, out_path)
    print(f"[+] Saved with NCHW input: {out_path}")

nhwc_to_nchw("onnx_out/efficientdet_lite0.onnx",
             "onnx_out/efficientdet_lite0_nchw.onnx")

nhwc_to_nchw("onnx_out/efficientdet_lite0_sim.onnx",
             "onnx_out/efficientdet_lite0_sim_nchw.onnx")

