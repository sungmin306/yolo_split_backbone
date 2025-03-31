import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
from pathlib import Path

def create_backbone_module(full_model, backbone_len=10):
    """
    전체 모델에서 앞쪽 backbone_len개의 레이어를 추출해 backbone 모듈 생성
    """
    backbone_layers = full_model.model[:backbone_len]

    class BackboneModule(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

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

    return BackboneModule(backbone_layers)

def create_neck_head_module(full_model, backbone_len=10, feature_indices=[4, 6, 9]):
    """
    전체 모델에서 backbone 이후의 레이어(neck 및 detect head)를 추출해 모듈 생성.
    YOLOv5 detect head는 3개의 피처(채널 64, 128, 256)를 입력받도록 설계되어 있으므로,
    backbone_outputs에서 feature_indices에 해당하는 피처들만 선택하여 전달합니다.
    """
    neck_head_layers = full_model.model[backbone_len:]
    class NeckHeadModule(nn.Module):
        def __init__(self, layers, feature_indices):
            super().__init__()
            self.layers = layers
            self.feature_indices = feature_indices

        def forward(self, backbone_outputs):
            # backbone_outputs가 전체(10개)라면 지정 인덱스만 선택하고,
            # 이미 선택된 경우엔 그대로 사용.
            if len(backbone_outputs) > len(self.feature_indices):
                inputs = tuple(backbone_outputs[i] for i in self.feature_indices)
            else:
                inputs = tuple(backbone_outputs)
            # detect head는 tuple 형태의 입력을 기대하므로 그대로 전달합니다.
            detection = self.layers(inputs)
            return detection
    return NeckHeadModule(neck_head_layers, feature_indices)

# ONNX export 시 wrapper 클래스들
class BackboneONNXWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        outs = self.backbone(x)
        return tuple(outs)

class NeckHeadONNXWrapper(nn.Module):
    def __init__(self, neck_head):
        super().__init__()
        self.neck_head = neck_head

    def forward(self, *selected_backbone_outputs):
        # selected_backbone_outputs는 이미 tuple 형태의 feature들 (예: 3개의 텐서)
        detection = self.neck_head(selected_backbone_outputs)
        return detection

if __name__ == "__main__":
    YOLO_ROOT = Path.cwd()
    model_path = YOLO_ROOT / 'yolov5n.pt'
    image_path = YOLO_ROOT / 'data/images/bus.jpg'

    # 모델 로드
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    full_model = ckpt['model'].float().eval()

    backbone_len = 10
    backbone = create_backbone_module(full_model, backbone_len=backbone_len).eval()

    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        backbone_outputs = backbone(input_tensor)
    print(f"Backbone PyTorch outputs count = {len(backbone_outputs)}")
    for i, o in enumerate(backbone_outputs):
        print(f" - outputs[{i}] shape = {list(o.shape)}")

    # 백본 ONNX 내보내기
    backbone_onnx = BackboneONNXWrapper(backbone).eval()
    onnx_backbone_path = YOLO_ROOT / "backbone.onnx"
    torch.onnx.export(
        backbone_onnx,
        input_tensor,
        str(onnx_backbone_path),
        input_names=['images'],
        output_names=[f'out{i}' for i in range(len(backbone_outputs))],
        opset_version=11
    )
    print(f"Backbone ONNX 모델이 저장되었습니다: {onnx_backbone_path}")

    # neck-head 모듈 생성 및 ONNX 내보내기
    selected_indices = [4, 6, 9]  # YOLOv5n에서 Detect head가 기대하는 피처 (채널 64,128,256)
    neck_head = create_neck_head_module(full_model, backbone_len=backbone_len, feature_indices=selected_indices).eval()
    # dummy 입력: backbone_outputs 중 선택된 피처들
    dummy_inputs = tuple([backbone_outputs[i] for i in selected_indices])
    neck_head_onnx = NeckHeadONNXWrapper(neck_head).eval()
    with torch.no_grad():
        dummy_detection = neck_head(dummy_inputs)
    onnx_neck_head_path = YOLO_ROOT / "neck-head.onnx"
    torch.onnx.export(
        neck_head_onnx,
        dummy_inputs,
        str(onnx_neck_head_path),
        input_names=[f'backbone_out{i}' for i in range(len(dummy_inputs))],
        output_names=["detection"],
        opset_version=11
    )
    print(f"Neck-Head ONNX 모델이 저장되었습니다: {onnx_neck_head_path}")
