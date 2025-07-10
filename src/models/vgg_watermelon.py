"""
VGG-16 Based Watermelon Sweetness Prediction Model
수박 당도 예측을 위한 VGG-16 기반 CNN 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


class VGGWatermelon(nn.Module):
    """
    VGG-16 기반 수박 당도 예측 모델
    
    멜-스펙트로그램 입력을 받아 당도값을 회귀 예측하는 모델
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 freeze_features: bool = False,
                 num_fc_layers: int = 2,
                 fc_hidden_size: int = 512):
        """
        VGGWatermelon 모델 초기화
        
        Args:
            input_channels (int): 입력 채널 수 (기본값: 3 - RGB)
            pretrained (bool): 사전 훈련된 가중치 사용 여부
            dropout_rate (float): 드롭아웃 비율
            freeze_features (bool): 특성 추출 레이어 고정 여부
            num_fc_layers (int): 완전 연결 레이어 수
            fc_hidden_size (int): 완전 연결 레이어 히든 크기
        """
        super(VGGWatermelon, self).__init__()
        
        self.input_channels = input_channels
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_features = freeze_features
        self.num_fc_layers = num_fc_layers
        self.fc_hidden_size = fc_hidden_size
        
        # VGG-16 백본 로드
        self.backbone = models.vgg16(pretrained=pretrained)
        
        # 입력 채널이 3이 아닌 경우 첫 번째 레이어 수정
        if input_channels != 3:
            self._modify_input_layer()
        
        # 특성 추출 레이어 고정
        if freeze_features:
            self._freeze_features()
        
        # 분류기 교체 (회귀용)
        self._build_regression_head()
        
    def _modify_input_layer(self):
        """입력 채널에 맞게 첫 번째 컨볼루션 레이어 수정"""
        # 첫 번째 레이어가 Conv2d인지 확인
        features_list = list(self.backbone.features.children())
        if not isinstance(features_list[0], nn.Conv2d):
            raise ValueError("첫 번째 레이어가 Conv2d가 아닙니다.")
        
        original_conv = features_list[0]
        
        # 새로운 첫 번째 레이어 생성
        new_conv = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # 가중치 초기화
        if self.pretrained and self.input_channels == 1:
            # 그레이스케일인 경우 RGB 가중치의 평균 사용
            with torch.no_grad():
                new_conv.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
        elif self.pretrained:
            # 다른 채널 수인 경우 적절히 초기화
            with torch.no_grad():
                if self.input_channels < 3:
                    new_conv.weight[:, :self.input_channels, :, :] = \
                        original_conv.weight[:, :self.input_channels, :, :]
                else:
                    # 채널 수가 더 많은 경우 반복해서 초기화
                    for i in range(self.input_channels):
                        new_conv.weight[:, i, :, :] = original_conv.weight[:, i % 3, :, :]
        
        # 백본의 첫 번째 레이어 교체
        # Sequential의 첫 번째 요소를 교체
        new_features = nn.Sequential(
            new_conv,
            *list(self.backbone.features.children())[1:]
        )
        self.backbone.features = new_features
    
    def _freeze_features(self):
        """특성 추출 레이어 고정"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
        # 어댑티브 풀링 레이어도 고정
        for param in self.backbone.avgpool.parameters():
            param.requires_grad = False
    
    def _build_regression_head(self):
        """회귀용 분류기 헤드 구성"""
        # 원본 분류기 제거
        classifier_list = list(self.backbone.classifier.children())
        if not isinstance(classifier_list[0], nn.Linear):
            raise ValueError("첫 번째 분류기 레이어가 Linear가 아닙니다.")
        
        in_features = classifier_list[0].in_features
        
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
        
        # 최종 출력층 (당도 예측)
        layers.append(nn.Linear(current_size, 1))
        
        # 분류기 교체
        self.backbone.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x (torch.Tensor): 입력 텐서 [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: 당도 예측값 [batch_size, 1]
        """
        # 입력 크기 검증
        if x.dim() != 4:
            raise ValueError(f"입력은 4차원 텐서여야 합니다. 현재: {x.dim()}차원")
        
        if x.size(1) != self.input_channels:
            raise ValueError(f"입력 채널 수가 맞지 않습니다. 예상: {self.input_channels}, 현재: {x.size(1)}")
        
        # VGG-16 순전파
        output = self.backbone(x)
        
        return output
    
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
            'model_name': 'VGGWatermelon',
            'backbone': 'VGG-16',
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


def create_vgg_watermelon(config_path: Optional[str] = None, **kwargs) -> VGGWatermelon:
    """
    설정 파일 또는 키워드 인자로부터 VGGWatermelon 모델 생성
    
    Args:
        config_path (str, optional): 모델 설정 파일 경로
        **kwargs: 모델 파라미터 (config_path보다 우선)
        
    Returns:
        VGGWatermelon: 생성된 모델
    """
    # 기본 설정
    default_config = {
        'input_channels': 3,
        'pretrained': True,
        'dropout_rate': 0.5,
        'freeze_features': False,
        'num_fc_layers': 2,
        'fc_hidden_size': 512
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
    
    # 모델 생성
    model = VGGWatermelon(**default_config)
    
    print(f"✅ VGGWatermelon 모델 생성 완료")
    if config_path:
        print(f"   📄 설정 파일: {config_path}")
    
    return model


def load_model_checkpoint(checkpoint_path: str, 
                         model_config: Optional[Dict[str, Any]] = None) -> VGGWatermelon:
    """
    체크포인트에서 모델 로드
    
    Args:
        checkpoint_path (str): 체크포인트 파일 경로
        model_config (dict, optional): 모델 설정 (체크포인트에 없는 경우)
        
    Returns:
        VGGWatermelon: 로드된 모델
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트 파일이 없습니다: {checkpoint_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 모델 설정 추출
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
    elif model_config is not None:
        config = model_config
    else:
        raise ValueError("모델 설정이 체크포인트에 없고 별도로 제공되지 않았습니다.")
    
    # 모델 생성 및 가중치 로드
    model = VGGWatermelon(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ 모델 체크포인트 로드 완료: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"   📊 에포크: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"   📉 손실: {checkpoint['loss']:.4f}")
    
    return model


def save_model_checkpoint(model: VGGWatermelon, 
                         save_path: str, 
                         epoch: Optional[int] = None,
                         loss: Optional[float] = None,
                         optimizer_state: Optional[Dict] = None,
                         **extra_info):
    """
    모델 체크포인트 저장
    
    Args:
        model (VGGWatermelon): 저장할 모델
        save_path (str): 저장 경로
        epoch (int, optional): 현재 에포크
        loss (float, optional): 현재 손실
        optimizer_state (dict, optional): 옵티마이저 상태
        **extra_info: 추가 정보
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 저장할 정보 구성
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model.get_model_info()
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    # 추가 정보 저장
    checkpoint.update(extra_info)
    
    # 체크포인트 저장
    torch.save(checkpoint, save_path)
    
    print(f"💾 모델 체크포인트 저장 완료: {save_path}")


if __name__ == "__main__":
    # 모델 테스트
    print("🧪 VGGWatermelon 모델 테스트")
    
    # 기본 모델 생성
    model = create_vgg_watermelon()
    model.print_model_info()
    
    # 테스트 입력
    batch_size = 4
    channels = 3
    height, width = 224, 224
    
    test_input = torch.randn(batch_size, channels, height, width)
    
    # 순전파 테스트
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        print(f"\n✅ 순전파 테스트 성공")
        print(f"   입력 크기: {test_input.shape}")
        print(f"   출력 크기: {output.shape}")
        print(f"   예측 당도: {output.squeeze().tolist()}") 