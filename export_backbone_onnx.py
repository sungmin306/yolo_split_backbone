import torch

import torch.nn as nn

import torchvision.transforms as T

import cv2

from pathlib import Path


def create_backbone_module(full_model, backbone_len=10):
    """

    full_model에서 앞쪽 backbone_len개 레이어만 추출해

    Skip-connection을 처리하는 BackboneModule을 생성

    """

    # YOLOv5의 model[i]는 nn.Module 형태 (CSP or conv 등)

    backbone_layers = full_model.model[:backbone_len]

    class BackboneModule(nn.Module):

        def __init__(self, layers):

            super().__init__()

            self.layers = layers  # nn.Sequential이 아님. YOLOv5는 skip 때문에 별도 forward필요

        def forward(self, x):

            outputs = []

            for m in self.layers:

                # m.f가 -1이 아닐 때(= skip이 있을 때) 이전 outputs 사용

                if m.f != -1:

                    if isinstance(m.f, int):

                        x = outputs[m.f]

                    else:

                        x = [outputs[j] for j in m.f]

                x = m(x)

                outputs.append(x)

                # Backbone 모든 레이어 출력(return)

            return outputs  # 리스트(각 레이어 output)

    return BackboneModule(backbone_layers)


if __name__ == "__main__":

    YOLO_ROOT = Path.cwd()

    model_path = YOLO_ROOT / 'yolov5n.pt'

    image_path = YOLO_ROOT / 'data/images/bus.jpg'

    # 전체 YOLOv5 모델 로드

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    full_model = ckpt['model'].float().eval()

    # Backbone module 생성

    backbone_len = 10

    backbone = create_backbone_module(full_model, backbone_len=backbone_len).eval()

    # 더미 입력 (640x640 이미지)

    img = cv2.imread(str(image_path))

    img = cv2.resize(img, (640, 640))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = T.Compose([T.ToTensor()])

    input_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 640, 640]

    # 1) PyTorch 버전으로 backbone 추론 테스트

    with torch.no_grad():

        outputs = backbone(input_tensor)

    print(f"Backbone PyTorch outputs count = {len(outputs)}")

    for i, o in enumerate(outputs):
        print(f" - outputs[{i}] shape = {list(o.shape)}")

        # 2) ONNX 변환


    # output이 "여러 텐서"이므로 onnx로 export할 때 outputs 수가 많아집니다.

    # (outputs를 튜플로 반환해야 합니다. list -> tuple 변환 필요)

    class BackboneONNXWrapper(nn.Module):

        def __init__(self, backbone):
            super().__init__()

            self.backbone = backbone

        def forward(self, x):
            outs = self.backbone(x)

            # onnx에서는 list로 반환이 어렵기 때문에 tuple로 변환

            return tuple(outs)


    backbone_onnx = BackboneONNXWrapper(backbone).eval()

    onnx_path = YOLO_ROOT / "backbone.onnx"

    torch.onnx.export(

        backbone_onnx,

        input_tensor,

        str(onnx_path),

        input_names=['images'],

        output_names=[f'out{i}' for i in range(len(outputs))],

        opset_version=11

    )

    print(f"Backbone ONNX 모델이 저장되었습니다: {onnx_path}")