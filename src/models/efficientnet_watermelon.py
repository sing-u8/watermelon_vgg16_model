"""
EfficientNet 기반 수박 당도 예측 모델
가장 효율적인 파라미터 대비 성능을 제공하는 모델
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


class EfficientNetWatermelon(nn.Module):
    """
    EfficientNet 기반 수박 당도 예측 모델
    
    작은 데이터셋에 최적화된 경량 모델
    """
    
    def __init__(self, 
                 model_name: str = 'efficientnet_b0',
                 input_channels: int = 3,
                 pretrained: bool = True,
                 dropout_rate: float = 0.7,
                 freeze_features: bool = False,
                 num_fc_layers: int = 2,
                 fc_hidden_size: int = 256):
        """
        EfficientNetWatermelon 모델 초기화
        
        Args:
            model_name (str): EfficientNet 모델명 (b0~b7)
            input_channels (int): 입력 채널 수
            pretrained (bool): 사전 훈련된 가중치 사용 여부
            dropout_rate (float): 드롭아웃 비율
            freeze_features (bool): 특성 추출 레이어 고정 여부
            num_fc_layers (int): 완전 연결 레이어 수
            fc_hidden_size (int): 완전 연결 레이어 히든 크기
        """
        super().__init__()
        
        self.model_name = model_name
        self.input_channels = input_channels
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_features = freeze_features
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        
        # EfficientNet 백본 로드
        self._load_backbone()
        
        # 입력 채널 수정
        if input_channels != 3:
            self._modify_input_layer()
        
        # 특성 추출 레이어 고정
        if freeze_features:
            self._freeze_features()
        
        # 분류기 교체 (회귀용)
        self._build_regression_head()
        
    def _load_backbone(self):
        """EfficientNet 백본 로드"""
        model_dict = {
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
            'efficientnet_b3': models.efficientnet_b3,
            'efficientnet_b4': models.efficientnet_b4,
            'efficientnet_b5': models.efficientnet_b5,
            'efficientnet_b6': models.efficientnet_b6,
            'efficientnet_b7': models.efficientnet_b7,
        }
        
        if self.model_name not in model_dict:
            raise ValueError(f"지원하지 않는 모델입니다: {self.model_name}")
        
        # PyTorch 버전 호환성을 위한 가중치 로드
        if self.pretrained:
            self.backbone = model_dict[self.model_name](weights='DEFAULT')
        else:
            self.backbone = model_dict[self.model_name](weights=None)
        
    def _modify_input_layer(self):
        """입력 채널에 맞게 첫 번째 레이어 수정"""
        # EfficientNet의 첫 번째 레이어 수정
        # 간단하게 3x3 conv로 교체
        new_conv = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=32,  # EfficientNet-B0 기본값
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        
        # 첫 번째 Conv2d 레이어 교체
        for i, module in enumerate(self.backbone.features[0]):
            if isinstance(module, nn.Conv2d):
                self.backbone.features[0][i] = new_conv
                break
        
    def _freeze_features(self):
        """특성 추출 레이어 고정"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
    def _build_regression_head(self):
        """회귀용 분류기 헤드 구성"""
        # 기존 분류기의 입력 차원 확인 (EfficientNet-B0: 1280)
        in_features = 1280  # EfficientNet-B0 기본값
        
        # 새로운 회귀 헤드 구성
        layers = []
        current_size = in_features
        
        # 여러 은닉층 추가
        for i in range(self.num_fc_layers - 1):
            layers.extend([
                nn.Linear(current_size, self.fc_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_rate)
            ])
            current_size = self.fc_hidden_size
        
        # 최종 출력층
        layers.append(nn.Linear(current_size, 1))
        
        # 분류기 교체
        self.backbone.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        if x.dim() != 4:
            raise ValueError(f"입력은 4차원 텐서여야 합니다. 현재: {x.dim()}차원")
        
        if x.size(1) != self.input_channels:
            raise ValueError(f"입력 채널 수가 맞지 않습니다. 예상: {self.input_channels}, 현재: {x.size(1)}")
        
        return self.backbone(x)
    
    def get_feature_maps(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        중간 특성 맵 추출 (시각화용)
        
        Args:
            x (torch.Tensor): 입력 텐서
            layer_idx (int, optional): 추출할 레이어 인덱스
            
        Returns:
            torch.Tensor: 특성 맵
        """
        if layer_idx is None:
            layer_idx = len(self.backbone.features) - 1
        
        with torch.no_grad():
            for i, layer in enumerate(self.backbone.features):
                x = layer(x)
                if i == layer_idx:
                    return x
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': f'EfficientNetWatermelon_{self.model_name}',
            'backbone': self.model_name,
            'input_channels': self.input_channels,
            'pretrained': self.pretrained,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_features': self.freeze_features,
            'dropout_rate': self.dropout_rate,
            'fc_layers': self.num_fc_layers,
            'fc_hidden_size': self.fc_hidden_size
        }
    
    def print_model_info(self):
        """모델 정보 출력"""
        info = self.get_model_info()
        print(f"🤖 {info['model_name']} 모델 정보")
        print(f"   📊 백본: {info['backbone']}")
        print(f"   🔧 입력 채널: {info['input_channels']}")
        print(f"   🎯 사전 훈련: {info['pretrained']}")
        print(f"   📈 총 파라미터: {info['total_parameters']:,}")
        print(f"   🔓 훈련 가능 파라미터: {info['trainable_parameters']:,}")
        print(f"   ❄️ 특성 고정: {info['frozen_features']}")
        print(f"   💧 드롭아웃: {info['dropout_rate']}")
        print(f"   🔗 FC 레이어: {info['fc_layers']}개")
        print(f"   📐 FC 히든 크기: {info['fc_hidden_size']}")


def create_efficientnet_watermelon(config_path: Optional[str] = None, **kwargs) -> EfficientNetWatermelon:
    """
    설정 파일 또는 키워드 인자로부터 EfficientNetWatermelon 모델 생성
    
    Args:
        config_path (str, optional): 모델 설정 파일 경로
        **kwargs: 모델 파라미터 (config_path보다 우선)
        
    Returns:
        EfficientNetWatermelon: 생성된 모델
    """
    # 기본 설정
    default_config = {
        'model_name': 'efficientnet_b0',
        'input_channels': 3,
        'pretrained': True,
        'dropout_rate': 0.7,
        'freeze_features': False,
        'num_fc_layers': 2,
        'fc_hidden_size': 256
    }
    
    # 설정 파일에서 로드
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config.get('model', {}))
    
    # 키워드 인자로 덮어쓰기
    default_config.update(kwargs)
    
    return EfficientNetWatermelon(**default_config)


def load_efficientnet_checkpoint(checkpoint_path: str, 
                                model_config: Optional[Dict[str, Any]] = None) -> EfficientNetWatermelon:
    """
    체크포인트에서 EfficientNet 모델 로드
    
    Args:
        checkpoint_path (str): 체크포인트 파일 경로
        model_config (dict, optional): 모델 설정 (없으면 체크포인트에서 로드)
        
    Returns:
        EfficientNetWatermelon: 로드된 모델
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 모델 설정 확인
    if model_config is None:
        model_config = checkpoint.get('model_config', {})
    
    # 모델 생성
    model = EfficientNetWatermelon(**model_config)
    
    # 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def save_efficientnet_checkpoint(model: EfficientNetWatermelon, 
                                save_path: str, 
                                epoch: Optional[int] = None,
                                loss: Optional[float] = None,
                                optimizer_state: Optional[Dict] = None,
                                **extra_info):
    """
    EfficientNet 모델 체크포인트 저장
    
    Args:
        model (EfficientNetWatermelon): 저장할 모델
        save_path (str): 저장 경로
        epoch (int, optional): 에포크 번호
        loss (float, optional): 손실 값
        optimizer_state (dict, optional): 옵티마이저 상태
        **extra_info: 추가 정보
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 체크포인트 데이터 구성
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_name': model.model_name,
            'input_channels': model.input_channels,
            'pretrained': model.pretrained,
            'dropout_rate': model.dropout_rate,
            'freeze_features': model.freeze_features,
            'num_fc_layers': model.num_fc_layers,
            'fc_hidden_size': model.fc_hidden_size
        },
        'model_info': model.get_model_info()
    }
    
    # 선택적 정보 추가
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    # 추가 정보 병합
    checkpoint.update(extra_info)
    
    # 저장
    torch.save(checkpoint, save_path)
    print(f"✅ EfficientNet 체크포인트 저장 완료: {save_path}")


# 사용 예시
if __name__ == "__main__":
    # 모델 생성 테스트
    model = create_efficientnet_watermelon()
    model.print_model_info()
    
    # 테스트 입력
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"출력 크기: {output.shape}")
    print(f"출력 값: {output.item():.4f}") 