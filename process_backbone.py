import torch
import cv2
from pathlib import Path
import requests
import io

YOLO_ROOT = Path.cwd()
image_path = YOLO_ROOT / 'data/images/bus.jpg'

# backbone_outputs.pt 파일 로드 (이미 split.py에서 생성한 파일)
backbone_outputs = torch.load(YOLO_ROOT / 'backbone_outputs.pt', map_location='cpu')

# 백본 출력 리스트를 메모리 버퍼에 저장 (바이너리 형식)
buffer = io.BytesIO()
torch.save(backbone_outputs, buffer)
buffer.seek(0)

# 서버에 전송
url = "http://localhost:8000/process_neck_head" # url = "http://10.0.5.56:30080/process_neck_head"
files = {"file": ("backbone_outputs.pt", buffer.getvalue())}
response = requests.post(url, files=files)
resp_json = response.json()
print("서버 응답:", resp_json)

# 서버에서 받은 검출 결과 (박스 좌표, 클래스, 신뢰도)
detections = resp_json.get("detections", [])

# 원본 이미지 로드 및 전처리 (결과 출력을 위한 이미지)
img = cv2.imread(str(image_path))
img_resized = cv2.resize(img, (640, 640))
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


# 결과 이미지 저장
# output_path = YOLO_ROOT / "output.jpg"
# cv2.imwrite(str(output_path), img_draw)
# print(f"결과 이미지가 {output_path} 에 저장되었습니다.")

cv2.imshow("Detections", img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
