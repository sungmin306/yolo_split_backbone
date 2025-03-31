import torch
import torchvision.transforms as T
import cv2
from pathlib import Path
import requests
import io
import time


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

# 안전한 globals를 추가하여 분할한 backbone 모델 로드 (weights_only=False 옵션 사용)
with torch.serialization.safe_globals({"__main__.BackboneModel": BackboneModel}):
    backbone_model = torch.load(backbone_model_path, map_location='cpu', weights_only=False).eval()

# 이미지 전처리 transform 정의 (640x640으로 resize)
transform = T.Compose([T.ToTensor()])

# 서버 URL (neck-head 처리를 위한 서버)
url = "http://10.0.5.56:30080/process_neck_head"

# 웹캠 열기 (기본 카메라 장치 0번 사용)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

print("실시간 카메라 스트림 시작 (5초마다 서버 전송)")

last_request_time = 0
detections = []  # 마지막 서버 전송에서 받은 검출 결과

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽어올 수 없습니다.")
        break

    # 프레임을 640x640으로 resize
    frame_resized = cv2.resize(frame, (640, 640))
    # BGR -> RGB 변환
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # tensor 변환 및 배치 차원 추가
    input_tensor = transform(frame_rgb).unsqueeze(0)  # [1, 3, 640, 640]

    # ▶ Backbone 추론 (실시간으로 진행)
    with torch.no_grad():
        backbone_outputs = backbone_model(input_tensor)

    # 5초마다 서버에 전송하여 neck-head 처리 수행
    current_time = time.time()
    if current_time - last_request_time >= 5:
        last_request_time = current_time
        # 백본 출력 리스트를 메모리 버퍼에 저장 (바이너리 형식)
        buffer = io.BytesIO()
        torch.save(backbone_outputs, buffer)
        buffer.seek(0)
        data_bytes = buffer.getvalue()
        data_size = len(data_bytes)
        print(f"전송 데이터 크기: {data_size} 바이트")

        # 서버에 전송 (동기 요청)
        try:
            files = {"file": ("backbone_outputs.pt", data_bytes)}
            response = requests.post(url, files=files, timeout=5)
            resp_json = response.json()
            detections = resp_json.get("detections", [])
            print("서버 응답:", detections)
        except Exception as e:
            print("서버 요청 실패:", e)
            detections = []

    # 프레임에 검출 결과 그리기 (마지막 서버 전송에서 받은 결과 사용)
    frame_draw = frame_resized.copy()  # BGR 이미지
    for det in detections:
        box = det["box"]
        label = det["class"]
        conf = det["confidence"]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_draw,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # 결과 프레임을 윈도우에 표시
    cv2.imshow("Detections", frame_draw)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 짧은 딜레이 (CPU 사용률 완화)
    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
