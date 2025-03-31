import torch
import torch.nn.functional as F
from pathlib import Path

# COCO 클래스 목록 (생략 – 기존 코드와 동일)
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

def expand_channels(x, target_channels):
    current_channels = x.shape[1]
    if current_channels < target_channels:
        pad_channels = target_channels - current_channels
        pad = torch.zeros((x.shape[0], pad_channels, x.shape[2], x.shape[3]),
                          dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)
    return x

def head_forward(layers, backbone_outputs):
    """
    YOLOv5 네크+헤드 forward 함수.
    backbone_outputs는 백본에서 반환된 리스트(인덱스 0~9).
    head_layers는 전체 모델의 뒷부분(인덱스 10 이후)입니다.
    """
    outputs = backbone_outputs.copy()  # 기존 백본 출력들을 포함
    x = outputs[-1]  # 백본의 마지막 출력 (인덱스 9)
    for m in layers:
        if m.f != -1:
            if isinstance(m.f, int):
                x = outputs[m.f]
            else:
                x = [outputs[j] for j in m.f]
        x = m(x)
        outputs.append(x)
    return outputs[-1]

device =torch.device("cuda" if torch.cuda.is_available() else "cpu") # torch.device("cuda") <- 이렇게 하니까 안됐음 쿠다 없을땐 CPU로 돌아가게끔

YOLO_ROOT = Path.cwd()
model_path = YOLO_ROOT / 'yolov5n.pt'
full_model = torch.load(model_path, map_location=device, weights_only=False)['model'].float().eval().to(device)

# 저장된 백본 출력 불러오기 (CUDA 디바이스로 로드)
backbone_outputs = torch.load("backbone_outputs.pt", map_location=device)
# 또는, 이미 로드된 backbone_outputs가 있다면 각 텐서를 device로 옮길 수 있음:
# backbone_outputs = [t.to(device) for t in backbone_outputs]

print(f"불러온 백본 출력 개수: {len(backbone_outputs)}, 마지막 출력 shape: {backbone_outputs[-1].shape}")

# backbone_len와 동일하게 head는 전체 모델의 뒷부분
backbone_len = 10
head_layers = full_model.model[backbone_len:]

with torch.no_grad():
    head_output = head_forward(head_layers, backbone_outputs)

# Detect 모듈이 튜플을 반환하는 경우, 첫 번째 요소를 사용
if isinstance(head_output, tuple):
    head_output = head_output[0]

print("\n[디텍션 결과] 추론 결과 확인")
print(type(head_output))
print(head_output.shape)



# head_output의 shape는 [1, num_dets, 85]로 가정
pred = head_output.squeeze(0)  # [num_dets, 85]
conf_thres = 0.20  # objectness threshold
mask = pred[:, 4] > conf_thres
filtered = pred[mask]

# 클래스 점수는 index 5: (85-5=80)
class_scores = filtered[:, 5:]
class_ids = class_scores.argmax(dim=1)
confidences = class_scores.max(dim=1).values

for i, (cls_id, conf) in enumerate(zip(class_ids, confidences)):
    label = CLASSES[int(cls_id)] if cls_id < len(CLASSES) else f"Unknown({cls_id})"
    print(f" [#{i}] Class: {label} ({conf:.2f})")
