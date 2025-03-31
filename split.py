import torch
import torchvision.transforms as T
import cv2
from pathlib import Path

def backbone_forward(layers, x):
    """YOLOv5 백본 forward 함수 (skip connection을 포함하여 각 레이어의 출력을 리스트에 저장)."""
    outputs = []
    for m in layers:
        # 만약 m.f가 -1이 아니라면 이전 출력들을 가져옴 (skip connection)
        if m.f != -1:
            if isinstance(m.f, int):
                x = outputs[m.f]
            else:
                x = [outputs[j] for j in m.f]
        x = m(x)
        outputs.append(x)
    return outputs

# 경로 설정
YOLO_ROOT = Path.cwd()
model_path = YOLO_ROOT / 'yolov5n.pt'
image_path = YOLO_ROOT / 'data/images/bus.jpg'

# 전체 모델 로드 (YOLOv5의 전체 모듈)
full_model = torch.load(model_path, map_location='cpu', weights_only=False)['model'].float().eval()

# YAML 기준: backbone은 10개의 레이어 (인덱스 0~9)
backbone_len = 10
backbone_layers = full_model.model[:backbone_len]

# 이미지 로딩 및 전처리 (여기서는 간단히 resize; 실제로는 letterbox 전처리 권장)
img = cv2.imread(str(image_path))
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
transform = T.Compose([T.ToTensor()])
input_tensor = transform(img).unsqueeze(0)  # [1, 3, 640, 640]

# ▶ Backbone 추론 (백본의 모든 중간 출력들을 얻음)
with torch.no_grad():
    backbone_outputs = backbone_forward(backbone_layers, input_tensor)

# 백본 출력(리스트)을 저장 (torch.save로 리스트 전체 저장)
torch.save(backbone_outputs, "backbone_outputs.pt")
print(f"저장 완료: backbone_outputs.pt, 출력 개수: {len(backbone_outputs)}")
