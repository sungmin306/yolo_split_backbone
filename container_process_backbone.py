import torch
import torchvision.transforms as T
import cv2
from pathlib import Path
import requests
import io

# BackboneModel 클래스 정의 (저장할 때 사용한 클래스)
class BackboneModel(torch.nn.Module):
    def __init__(self, layers):
        super(BackboneModel, self).__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        outputs = []
        for m in self.layers:
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
backbone_model_path = YOLO_ROOT / 'yolov5n_backbone.pt'
image_path = YOLO_ROOT / 'data/images/bus.jpg'

# 안전한 globals를 추가하여 분할한 backbone 모델 로드
with torch.serialization.safe_globals({"__main__.BackboneModel": BackboneModel}):
    backbone_model = torch.load(backbone_model_path, map_location='cpu', weights_only=False).eval()


# 이미지 로딩 및 전처리 (간단히 resize; 실제로는 letterbox 전처리 권장)
img = cv2.imread(str(image_path))
img_resized = cv2.resize(img, (640, 640))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
transform = T.Compose([T.ToTensor()])
input_tensor = transform(img_rgb).unsqueeze(0)  # [1, 3, 640, 640]

# ▶ Backbone 추론: 분할한 backbone 모델을 사용하여 중간 출력들을 얻음
with torch.no_grad():
    backbone_outputs = backbone_model(input_tensor)

# 백본 출력 리스트를 메모리 버퍼에 저장 (바이너리 형식)
buffer = io.BytesIO()
torch.save(backbone_outputs, buffer)
buffer.seek(0)

# 서버에 전송 (예: neck-head 처리를 위한 서버)
url = "http://10.0.5.56:30080/process_neck_head"
files = {"file": ("backbone_outputs.pt", buffer.getvalue())}
response = requests.post(url, files=files)
resp_json = response.json()
print("서버 응답:", resp_json)

# 서버에서 받은 검출 결과 (박스 좌표, 클래스, 신뢰도)
detections = resp_json.get("detections", [])

# 원본 이미지(또는 resize된 이미지)에 박스 그리기
img_draw = img_resized.copy()  # resize된 BGR 이미지
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

# 결과 이미지 저장
output_path = YOLO_ROOT / "output.jpg"
cv2.imwrite(str(output_path), img_draw)
print(f"결과 이미지가 {output_path} 에 저장되었습니다.")

cv2.imshow("Detections", img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
