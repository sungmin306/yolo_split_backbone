import cv2
import requests
import io
import torch
import onnxruntime
import numpy as np
from pathlib import Path

# 경로 설정
YOLO_ROOT = Path.cwd()
image_path = YOLO_ROOT / 'data/images/bus.jpg'
onnx_model_path = YOLO_ROOT / 'backbone.onnx'

# 1. 입력 이미지 전처리
img = cv2.imread(str(image_path))
img_resized = cv2.resize(img, (640, 640))
# OpenCV는 BGR이므로 RGB로 변환
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
# [H, W, C] -> [C, H, W] 및 [0,255] -> [0,1] 변환
img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
# 배치 차원 추가
input_tensor = img_tensor.unsqueeze(0)  # shape: [1, 3, 640, 640]
# numpy 형식의 입력 (onnxruntime는 numpy array 입력 사용)
input_numpy = input_tensor.numpy()

# 2. ONNX 모델을 이용해 추론 (backbone.onnx)
session = onnxruntime.InferenceSession(str(onnx_model_path))
# 입력 이름은 export 시 지정한 이름 "images" 또는 확인된 이름 사용
input_name = session.get_inputs()[0].name
# 출력 이름 목록 (예: out0, out1, ..., out9)
output_names = [o.name for o in session.get_outputs()]
onnx_outputs = session.run(output_names, {input_name: input_numpy})

# onnxruntime의 출력은 numpy 배열이므로 torch.Tensor로 변환
backbone_outputs_onnx = [torch.from_numpy(out) for out in onnx_outputs]
print(f"ONNX Backbone outputs count = {len(backbone_outputs_onnx)}")
for i, o in enumerate(backbone_outputs_onnx):
    print(f" - outputs[{i}] shape = {list(o.shape)}")

# 3. 백본 출력 데이터를 로컬 파일로 저장하는 기능 추가
def save_backbone_outputs(outputs, save_path):
    """백본 출력 데이터를 파일로 저장합니다."""
    torch.save(outputs, save_path)
    print(f"Backbone outputs saved at {save_path}")

save_path = YOLO_ROOT / "backbone_outputs_onnx.pt"
save_backbone_outputs(backbone_outputs_onnx, save_path)

# 4. 저장한 데이터를 메모리 버퍼로 불러와 서버로 전송
with open(save_path, "rb") as f:
    file_data = f.read()

buffer = io.BytesIO(file_data)
buffer.seek(0)

url = "http://localhost:8000/process_neck_head"  # 서버 URL
files = {"file": ("backbone_outputs_onnx.pt", buffer.getvalue())}

try:
    response = requests.post(url, files=files)
    response.raise_for_status()
    resp_json = response.json()
    print("서버 응답:", resp_json)
except Exception as e:
    print("서버 요청 중 오류 발생:", e)
    resp_json = {}

# 5. 서버로부터 받은 검출 결과로 원본 이미지에 박스 처리
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

detections = resp_json.get("detections", [])
print(f"총 {len(detections)} 개의 객체 검출됨.")

img_draw = img_resized.copy()
for det in detections:
    box = det["box"]
    label = det["class"]
    conf = det["confidence"]
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img_draw,
        f"{label} {conf:.2f}",
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

output_path = YOLO_ROOT / "output.jpg"
cv2.imwrite(str(output_path), img_draw)
print(f"결과 이미지가 {output_path} 에 저장되었습니다.")

cv2.imshow("Detections", img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
