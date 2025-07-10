"""
Loss Functions Module
수박 당도 예측을 위한 손실 함수 모듈
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np


class RegressionLoss(nn.Module):
    """
    회귀 문제를 위한 통합 손실 함수 클래스
    """
    
    def __init__(self, 
                 loss_type: str = 'mse',
                 reduction: str = 'mean',
                 huber_delta: float = 1.0,
                 quantile_alpha: float = 0.5,
                 weights: Optional[torch.Tensor] = None):
        """
        RegressionLoss 초기화
        
        Args:
            loss_type (str): 손실 함수 타입 ('mse', 'mae', 'huber', 'smooth_l1', 'quantile')
            reduction (str): 축소 방식 ('mean', 'sum', 'none')
            huber_delta (float): Huber 손실에서 사용할 델타 값
            quantile_alpha (float): Quantile 회귀에서 사용할 알파 값
            weights (torch.Tensor, optional): 샘플별 가중치
        """
        super(RegressionLoss, self).__init__()
        
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.huber_delta = huber_delta
        self.quantile_alpha = quantile_alpha
        self.weights = weights
        
        # 지원하는 손실 함수 타입
        self.supported_losses = ['mse', 'mae', 'huber', 'smooth_l1', 'quantile', 'mape']
        
        if self.loss_type not in self.supported_losses:
            raise ValueError(f"지원하지 않는 손실 함수: {loss_type}. "
                           f"지원되는 타입: {self.supported_losses}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        손실 계산
        
        Args:
            predictions (torch.Tensor): 모델 예측값 [batch_size, 1] 또는 [batch_size]
            targets (torch.Tensor): 실제 타겟값 [batch_size, 1] 또는 [batch_size]
            
        Returns:
            torch.Tensor: 계산된 손실
        """
        # 텐서 차원 정규화
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # 배치 크기 확인
        if predictions.shape != targets.shape:
            raise ValueError(f"예측값과 타겟값의 크기가 다릅니다: "
                           f"{predictions.shape} vs {targets.shape}")
        
        # 손실 함수별로 계산
        if self.loss_type == 'mse':
            loss = self._mse_loss(predictions, targets)
        elif self.loss_type == 'mae':
            loss = self._mae_loss(predictions, targets)
        elif self.loss_type == 'huber':
            loss = self._huber_loss(predictions, targets)
        elif self.loss_type == 'smooth_l1':
            loss = self._smooth_l1_loss(predictions, targets)
        elif self.loss_type == 'quantile':
            loss = self._quantile_loss(predictions, targets)
        elif self.loss_type == 'mape':
            loss = self._mape_loss(predictions, targets)
        else:
            raise ValueError(f"구현되지 않은 손실 함수: {self.loss_type}")
        
        # 가중치 적용
        if self.weights is not None:
            weights = self.weights.to(loss.device)
            if weights.shape[0] != loss.shape[0]:
                raise ValueError(f"가중치 크기가 배치 크기와 다릅니다: "
                               f"{weights.shape[0]} vs {loss.shape[0]}")
            loss = loss * weights
        
        # 축소 적용
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
    
    def _mse_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """평균 제곱 오차 (MSE) 계산"""
        return F.mse_loss(predictions, targets, reduction='none')
    
    def _mae_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """평균 절대 오차 (MAE) 계산"""
        return F.l1_loss(predictions, targets, reduction='none')
    
    def _huber_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Huber 손실 계산"""
        return F.huber_loss(predictions, targets, delta=self.huber_delta, reduction='none')
    
    def _smooth_l1_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Smooth L1 손실 계산"""
        return F.smooth_l1_loss(predictions, targets, reduction='none')
    
    def _quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Quantile 손실 계산"""
        residuals = targets - predictions
        loss = torch.max(
            self.quantile_alpha * residuals,
            (self.quantile_alpha - 1) * residuals
        )
        return loss
    
    def _mape_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """평균 절대 백분율 오차 (MAPE) 계산"""
        # 0으로 나누기 방지
        epsilon = 1e-8
        return torch.abs((targets - predictions) / (targets + epsilon)) * 100


class WeightedMSELoss(nn.Module):
    """
    가중치가 적용된 MSE 손실
    당도값에 따라 다른 가중치를 적용할 수 있음
    """
    
    def __init__(self, sweetness_ranges: Optional[Dict[str, tuple]] = None):
        """
        WeightedMSELoss 초기화
        
        Args:
            sweetness_ranges (Dict, optional): 당도 범위별 가중치
                예: {'low': (8.0, 10.0, 1.2), 'medium': (10.0, 11.5, 1.0), 'high': (11.5, 13.0, 1.5)}
        """
        super(WeightedMSELoss, self).__init__()
        
        if sweetness_ranges is None:
            # 기본 가중치 설정 (극값에 더 높은 가중치)
            self.sweetness_ranges = {
                'low': (8.0, 9.5, 1.2),      # 낮은 당도: 가중치 1.2
                'medium': (9.5, 11.0, 1.0),  # 중간 당도: 가중치 1.0
                'high': (11.0, 13.0, 1.3)    # 높은 당도: 가중치 1.3
            }
        else:
            self.sweetness_ranges = sweetness_ranges
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        가중치가 적용된 MSE 손실 계산
        
        Args:
            predictions (torch.Tensor): 모델 예측값
            targets (torch.Tensor): 실제 타겟값
            
        Returns:
            torch.Tensor: 가중치가 적용된 MSE 손실
        """
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # 기본 MSE 계산
        mse_loss = F.mse_loss(predictions, targets, reduction='none')
        
        # 당도값에 따른 가중치 계산
        weights = torch.ones_like(targets)
        
        for range_name, (min_val, max_val, weight) in self.sweetness_ranges.items():
            mask = (targets >= min_val) & (targets < max_val)
            weights[mask] = weight
        
        # 가중치 적용
        weighted_loss = mse_loss * weights
        
        return weighted_loss.mean()


class CombinedLoss(nn.Module):
    """
    여러 손실 함수를 조합한 손실 함수
    """
    
    def __init__(self, 
                 loss_configs: Dict[str, Dict[str, Any]],
                 loss_weights: Optional[Dict[str, float]] = None):
        """
        CombinedLoss 초기화
        
        Args:
            loss_configs (Dict): 손실 함수별 설정
                예: {'mse': {'reduction': 'mean'}, 'mae': {'reduction': 'mean'}}
            loss_weights (Dict, optional): 손실 함수별 가중치
                예: {'mse': 0.7, 'mae': 0.3}
        """
        super(CombinedLoss, self).__init__()
        
        self.loss_functions = nn.ModuleDict()
        self.loss_weights = loss_weights or {}
        
        # 손실 함수 생성
        for loss_name, config in loss_configs.items():
            if loss_name in ['mse', 'mae', 'huber', 'smooth_l1', 'quantile', 'mape']:
                self.loss_functions[loss_name] = RegressionLoss(
                    loss_type=loss_name, **config
                )
            elif loss_name == 'weighted_mse':
                self.loss_functions[loss_name] = WeightedMSELoss(**config)
            else:
                raise ValueError(f"지원하지 않는 손실 함수: {loss_name}")
        
        # 기본 가중치 설정 (균등 분배)
        if not self.loss_weights:
            num_losses = len(self.loss_functions)
            self.loss_weights = {name: 1.0/num_losses for name in self.loss_functions.keys()}
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        조합된 손실 계산
        
        Args:
            predictions (torch.Tensor): 모델 예측값
            targets (torch.Tensor): 실제 타겟값
            
        Returns:
            torch.Tensor: 조합된 손실
        """
        total_loss = None
        
        for loss_name, loss_fn in self.loss_functions.items():
            loss_value = loss_fn(predictions, targets)
            weight = self.loss_weights.get(loss_name, 1.0)
            weighted_loss = weight * loss_value
            
            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss
        
        return total_loss if total_loss is not None else torch.tensor(0.0)


def create_loss_function(loss_config: Dict[str, Any]) -> nn.Module:
    """
    설정에 따른 손실 함수 생성
    
    Args:
        loss_config (Dict): 손실 함수 설정
        
    Returns:
        nn.Module: 생성된 손실 함수
    """
    loss_type = loss_config.get('type', 'mse').lower()
    
    if loss_type in ['mse', 'mae', 'huber', 'smooth_l1', 'quantile', 'mape']:
        return RegressionLoss(
            loss_type=loss_type,
            reduction=loss_config.get('reduction', 'mean'),
            huber_delta=loss_config.get('huber_delta', 1.0),
            quantile_alpha=loss_config.get('quantile_alpha', 0.5)
        )
    
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(
            sweetness_ranges=loss_config.get('sweetness_ranges')
        )
    
    elif loss_type == 'combined':
        return CombinedLoss(
            loss_configs=loss_config.get('loss_configs', {}),
            loss_weights=loss_config.get('loss_weights')
        )
    
    else:
        raise ValueError(f"지원하지 않는 손실 함수 타입: {loss_type}")


def calculate_loss_weights(targets: torch.Tensor, 
                          num_bins: int = 5,
                          strategy: str = 'inverse_frequency') -> torch.Tensor:
    """
    타겟값 분포에 따른 손실 가중치 계산
    
    Args:
        targets (torch.Tensor): 타겟값들
        num_bins (int): 구간 수
        strategy (str): 가중치 계산 전략 ('inverse_frequency', 'balanced')
        
    Returns:
        torch.Tensor: 계산된 가중치
    """
    # 히스토그램 계산
    min_val, max_val = targets.min().item(), targets.max().item()
    bins = torch.linspace(min_val, max_val, num_bins + 1)
    
    # 각 샘플이 속한 구간 찾기
    bin_indices = torch.searchsorted(bins[1:], targets)
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
    
    # 구간별 샘플 수 계산
    bin_counts = torch.bincount(bin_indices, minlength=num_bins).float()
    
    # 가중치 계산
    if strategy == 'inverse_frequency':
        # 빈도의 역수로 가중치 계산
        weights_per_bin = 1.0 / (bin_counts + 1e-8)
    elif strategy == 'balanced':
        # 균형 가중치 계산
        total_samples = len(targets)
        weights_per_bin = total_samples / (num_bins * bin_counts + 1e-8)
    else:
        raise ValueError(f"지원하지 않는 가중치 전략: {strategy}")
    
    # 정규화
    weights_per_bin = weights_per_bin / weights_per_bin.mean()
    
    # 각 샘플에 가중치 할당
    sample_weights = weights_per_bin[bin_indices]
    
    return sample_weights


def smoothed_loss(predictions: torch.Tensor, 
                  targets: torch.Tensor,
                  base_loss_fn: nn.Module,
                  smoothing_factor: float = 0.1) -> torch.Tensor:
    """
    라벨 스무딩이 적용된 손실 함수
    
    Args:
        predictions (torch.Tensor): 모델 예측값
        targets (torch.Tensor): 실제 타겟값
        base_loss_fn (nn.Module): 기본 손실 함수
        smoothing_factor (float): 스무딩 강도
        
    Returns:
        torch.Tensor: 스무딩이 적용된 손실
    """
    # 기본 손실 계산
    base_loss = base_loss_fn(predictions, targets)
    
    # 균등 분포와의 KL divergence 추가
    # 회귀 문제에서는 타겟값 주변의 가우시안 분포로 스무딩
    std = torch.std(targets)
    gaussian_noise = torch.randn_like(targets) * std * smoothing_factor
    smoothed_targets = targets + gaussian_noise
    
    smooth_loss = base_loss_fn(predictions, smoothed_targets)
    
    # 조합
    return (1 - smoothing_factor) * base_loss + smoothing_factor * smooth_loss


if __name__ == "__main__":
    # 손실 함수 테스트
    print("🧪 손실 함수 테스트")
    
    # 테스트 데이터 생성
    batch_size = 32
    predictions = torch.randn(batch_size, 1) * 2 + 10  # 8~12 범위
    targets = torch.randn(batch_size, 1) * 1.5 + 10.5  # 실제 당도값
    
    print(f"📊 테스트 데이터: 예측값 {predictions.shape}, 타겟값 {targets.shape}")
    
    # 다양한 손실 함수 테스트
    loss_types = ['mse', 'mae', 'huber', 'smooth_l1']
    
    for loss_type in loss_types:
        loss_fn = RegressionLoss(loss_type=loss_type)
        loss_value = loss_fn(predictions, targets)
        print(f"   {loss_type.upper()}: {loss_value.item():.4f}")
    
    # 가중치 적용 MSE 테스트
    weighted_mse = WeightedMSELoss()
    weighted_loss = weighted_mse(predictions, targets)
    print(f"   Weighted MSE: {weighted_loss.item():.4f}")
    
    # 조합 손실 테스트
    combined_config = {
        'mse': {'reduction': 'mean'},
        'mae': {'reduction': 'mean'}
    }
    combined_weights = {'mse': 0.7, 'mae': 0.3}
    combined_loss = CombinedLoss(combined_config, combined_weights)
    combined_value = combined_loss(predictions, targets)
    print(f"   Combined (MSE+MAE): {combined_value.item():.4f}")
    
    print("✅ 손실 함수 테스트 완료") 