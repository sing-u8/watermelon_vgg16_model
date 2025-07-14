"""
Mel-Spectrogram CNN 기반 수박 당도 예측 모델
오디오 도메인에 특화된 경량 CNN 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


class MelSpecCNNWatermelon(nn.Module):
    """
    Mel-Spectrogram CNN 기반 수박 당도 예측 모델
    
    오디오 멜-스펙트로그램 처리에 특화된 경량 모델
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 dropout_rate: float = 0.7,
                 num_classes: int = 1,
                 base_channels: int = 32):
        """
        MelSpecCNNWatermelon 모델 초기화
        
        Args:
            input_channels (int): 입력 채널 수
            dropout_rate (float): 드롭아웃 비율
            num_classes (int): 출력 클래스 수 (회귀의 경우 1)
            base_channels (int): 기본 채널 수
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.base_channels = base_channels
        
        # 특성 추출 레이어들
        self.features = self._build_feature_layers()
        
        # 분류기 레이어들
        self.classifier = self._build_classifier()
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _build_feature_layers(self):
        """특성 추출 레이어 구성"""
        layers = []
        
        # 첫 번째 블록
        layers.extend([
            nn.Conv2d(self.input_channels, self.base_channels, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # 두 번째 블록
        layers.extend([
            nn.Conv2d(self.base_channels, self.base_channels * 2, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # 세 번째 블록
        layers.extend([
            nn.Conv2d(self.base_channels * 2, self.base_channels * 4, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # 네 번째 블록
        layers.extend([
            nn.Conv2d(self.base_channels * 4, self.base_channels * 8, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # 다섯 번째 블록 (더 깊은 특성 추출)
        layers.extend([
            nn.Conv2d(self.base_channels * 8, self.base_channels * 16, 
                     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.base_channels * 16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        ])
        
        return nn.Sequential(*layers)
    
    def _build_classifier(self):
        """분류기 레이어 구성"""
        return nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.base_channels * 16, self.base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.base_channels * 4, self.base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.base_channels, self.num_classes)
        )
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
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
        
        # 특성 추출
        features = self.features(x)
        
        # 평탄화
        features = features.view(features.size(0), -1)
        
        # 분류
        output = self.classifier(features)
        
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
            layer_idx = len(self.features) - 1
        
        with torch.no_grad():
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == layer_idx:
                    return x
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MelSpecCNNWatermelon',
            'backbone': 'Custom Mel-Spectrogram CNN',
            'input_channels': self.input_channels,
            'base_channels': self.base_channels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate,
            'num_classes': self.num_classes
        }
    
    def print_model_info(self):
        """모델 정보 출력"""
        info = self.get_model_info()
        print(f"🤖 {info['model_name']} 모델 정보")
        print(f"   📊 백본: {info['backbone']}")
        print(f"   🔧 입력 채널: {info['input_channels']}")
        print(f"   🎯 기본 채널: {info['base_channels']}")
        print(f"   📈 총 파라미터: {info['total_parameters']:,}")
        print(f"   🔓 훈련 가능 파라미터: {info['trainable_parameters']:,}")
        print(f"   💧 드롭아웃: {info['dropout_rate']}")
        print(f"   🎯 출력 클래스: {info['num_classes']}")


def create_melspec_cnn_watermelon(config_path: Optional[str] = None, **kwargs) -> MelSpecCNNWatermelon:
    """
    설정 파일 또는 키워드 인자로부터 MelSpecCNNWatermelon 모델 생성
    
    Args:
        config_path (str, optional): 모델 설정 파일 경로
        **kwargs: 모델 파라미터 (config_path보다 우선)
        
    Returns:
        MelSpecCNNWatermelon: 생성된 모델
    """
    # 기본 설정
    default_config = {
        'input_channels': 3,
        'dropout_rate': 0.7,
        'num_classes': 1,
        'base_channels': 32
    }
    
    # 설정 파일에서 로드
    if config_path:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config.get('model', {}))
        except FileNotFoundError:
            print(f"⚠️ 설정 파일을 찾을 수 없습니다: {config_path}")
    
    # 키워드 인자로 덮어쓰기
    default_config.update(kwargs)
    
    return MelSpecCNNWatermelon(**default_config)


def load_melspec_cnn_checkpoint(checkpoint_path: str, 
                                model_config: Optional[Dict[str, Any]] = None) -> MelSpecCNNWatermelon:
    """
    체크포인트에서 MelSpecCNN 모델 로드
    
    Args:
        checkpoint_path (str): 체크포인트 파일 경로
        model_config (dict, optional): 모델 설정 (없으면 체크포인트에서 로드)
        
    Returns:
        MelSpecCNNWatermelon: 로드된 모델
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 모델 설정 확인
    if model_config is None:
        model_config = checkpoint.get('model_config', {})
    
    # 모델 생성
    model = MelSpecCNNWatermelon(**model_config)
    
    # 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def save_melspec_cnn_checkpoint(model: MelSpecCNNWatermelon, 
                                save_path: str, 
                                epoch: Optional[int] = None,
                                loss: Optional[float] = None,
                                optimizer_state: Optional[Dict] = None,
                                **extra_info):
    """
    MelSpecCNN 모델 체크포인트 저장
    
    Args:
        model (MelSpecCNNWatermelon): 저장할 모델
        save_path (str): 저장 경로
        epoch (int, optional): 에포크 번호
        loss (float, optional): 손실 값
        optimizer_state (dict, optional): 옵티마이저 상태
        **extra_info: 추가 정보
    """
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 체크포인트 데이터 구성
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_channels': model.input_channels,
            'dropout_rate': model.dropout_rate,
            'num_classes': model.num_classes,
            'base_channels': model.base_channels
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
    print(f"✅ MelSpecCNN 체크포인트 저장 완료: {save_path}")


# 사용 예시
if __name__ == "__main__":
    # 모델 생성 테스트
    model = create_melspec_cnn_watermelon()
    model.print_model_info()
    
    # 테스트 입력
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"출력 크기: {output.shape}")
    print(f"출력 값: {output.item():.4f}")
    
    # 특성 맵 추출 테스트
    feature_maps = model.get_feature_maps(test_input, layer_idx=5)
    print(f"특성 맵 크기: {feature_maps.shape}") 