import torch
import torch.nn as nn
from pathlib import Path

class BackboneModel(nn.Module):
    """
    전체 모델에서 추출한 backbone 레이어(앞쪽 10개)를 ModuleList에 저장하여 forward 시
    skip connection 처리를 수행하는 모듈.
    """
    def __init__(self, layers):
        super(BackboneModel, self).__init__()
        # layers를 nn.ModuleList에 넣어서, 저장 및 로드 시 자동으로 관리되게 함.
        self.layers = nn.ModuleList(layers)

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

if __name__ == "__main__":
    # 경로 설정
    YOLO_ROOT = Path.cwd()
    model_path = YOLO_ROOT / 'yolov5n.pt'
    backbone_model_path = YOLO_ROOT / 'yolov5n_backbone.pt'

    # 전체 모델 로드 (전체 모델 내 'model' 키 하위의 모듈 리스트를 사용)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    full_model = ckpt['model'].float().eval()

    # YAML 기준: backbone은 앞쪽 10개 레이어 (인덱스 0~9)
    backbone_layers = full_model.model[:10]

    # BackboneModel 인스턴스 생성
    backbone_model = BackboneModel(backbone_layers).eval()

    # Backbone 모델 저장 (전체 모델 객체로 저장하는 경우)
    torch.save(backbone_model, backbone_model_path)
    print(f"Backbone 모델이 {backbone_model_path} 에 저장되었습니다.")

    # 또는 state_dict만 저장할 수도 있음
    # torch.save(backbone_model.state_dict(), backbone_model_path)
    # print(f"Backbone 모델의 state_dict가 {backbone_model_path} 에 저장되었습니다.")
