from pathlib import Path 

import torch 

import cv2 

import numpy as np 

 

# 경로 설정 

YOLO_ROOT = Path.cwd() 

model_path = YOLO_ROOT / 'yolov5n.pt' 

image_path = YOLO_ROOT / 'data/images/bus.jpg' 

 

# 전체 모델 로드 (torch.hub로 로드 권장) 

model = torch.hub.load(YOLO_ROOT, 'custom', path=model_path, source='local') 

 

# 이미지 로드 (BGR -> RGB로 변환) 

img = cv2.imread(str(image_path)) 

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

 

# 객체 탐지 실행 (모델에 넘기기 전에 이미지가 numpy여도 자동으로 처리됨) 

results = model(img_rgb) 

 

# 탐지 결과 출력 

print(results.pandas().xyxy[0]) 

 

# 결과 시각화 

results.render() 

result_img = results.ims[0] 

 

# 결과 이미지 표시 (OpenCV 창) 

cv2.imshow('YOLOv5 Detection', result_img) 

cv2.waitKey(0) 

cv2.destroyAllWindows()


