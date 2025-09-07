```bash
# (권장) Python 3.10 가상환경
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 필요 패키지
pip install "tf2onnx>=1.16.1" "tensorflow>=2.12,<2.16" onnx onnxruntime onnxsim numpy requests

# 실행
python3 convert.py
```
