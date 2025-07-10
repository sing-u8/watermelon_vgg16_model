"""
Trainer Module
수박 당도 예측 모델 훈련을 위한 통합 트레이너 모듈
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from typing import Dict, Any, Optional, List, Tuple
import yaml
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import warnings

# 상대 import
import sys
from pathlib import Path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from models.vgg_watermelon import VGGWatermelon, save_model_checkpoint
from training.data_loader import WatermelonDataLoader
from training.loss_functions import create_loss_function
from training.metrics import RegressionMetrics, MetricsTracker


class WatermelonTrainer:
    """
    수박 당도 예측 모델 훈련을 위한 통합 트레이너 클래스
    """
    
    def __init__(self, 
                 model: VGGWatermelon,
                 data_loader: WatermelonDataLoader,
                 loss_function: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 save_dir: str = "experiments",
                 experiment_name: str = "watermelon_training"):
        """
        WatermelonTrainer 초기화
        
        Args:
            model (VGGWatermelon): 훈련할 모델
            data_loader (WatermelonDataLoader): 데이터 로더
            loss_function (nn.Module): 손실 함수
            optimizer (optim.Optimizer): 옵티마이저
            device (torch.device): 디바이스
            scheduler (optim.lr_scheduler._LRScheduler, optional): 학습률 스케줄러
            save_dir (str): 모델 저장 디렉토리
            experiment_name (str): 실험 이름
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        # 저장 경로 설정
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 메트릭 추적
        self.metrics_tracker = MetricsTracker()
        
        # 훈련 상태
        self.current_epoch = 0
        self.best_val_mae = float('inf')
        self.best_model_path = self.save_dir / "best_model.pth"
        self.last_model_path = self.save_dir / "last_model.pth"
        
        # Early stopping
        self.early_stopping_patience = 10
        self.early_stopping_counter = 0
        
        print(f"🚀 WatermelonTrainer 초기화 완료")
        print(f"   💾 저장 경로: {self.save_dir}")
        print(f"   🖥️ 디바이스: {self.device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        한 에포크 훈련
        
        Returns:
            Dict[str, float]: 훈련 메트릭
        """
        self.model.train()
        
        # 메트릭 초기화
        train_metrics = RegressionMetrics(self.device)
        epoch_loss = 0.0
        num_batches = 0
        
        # 진행 바 설정
        pbar = tqdm(self.data_loader.train_loader, 
                   desc=f"Epoch {self.current_epoch+1} [Train]",
                   leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # 데이터를 디바이스로 이동
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 그래디언트 초기화
            self.optimizer.zero_grad()
            
            # 순전파
            predictions = self.model(inputs)
            
            # 손실 계산
            loss = self.loss_function(predictions, targets)
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑 (선택적)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 옵티마이저 스텝
            self.optimizer.step()
            
            # 메트릭 업데이트
            train_metrics.update(predictions, targets, loss)
            epoch_loss += loss.item()
            num_batches += 1
            
            # 진행 바 업데이트
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{epoch_loss/num_batches:.4f}"
            })
        
        # 훈련 메트릭 계산
        metrics = train_metrics.compute()
        metrics['avg_loss'] = epoch_loss / num_batches
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        한 에포크 검증
        
        Returns:
            Dict[str, float]: 검증 메트릭
        """
        self.model.eval()
        
        # 메트릭 초기화
        val_metrics = RegressionMetrics(self.device)
        epoch_loss = 0.0
        num_batches = 0
        
        # 진행 바 설정
        pbar = tqdm(self.data_loader.val_loader, 
                   desc=f"Epoch {self.current_epoch+1} [Val]",
                   leave=False)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(pbar):
                # 데이터를 디바이스로 이동
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 순전파
                predictions = self.model(inputs)
                
                # 손실 계산
                loss = self.loss_function(predictions, targets)
                
                # 메트릭 업데이트
                val_metrics.update(predictions, targets, loss)
                epoch_loss += loss.item()
                num_batches += 1
                
                # 진행 바 업데이트
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Avg Loss': f"{epoch_loss/num_batches:.4f}"
                })
        
        # 검증 메트릭 계산
        metrics = val_metrics.compute()
        metrics['avg_loss'] = epoch_loss / num_batches
        
        return metrics
    
    def test_model(self) -> Dict[str, float]:
        """
        모델 테스트
        
        Returns:
            Dict[str, float]: 테스트 메트릭
        """
        self.model.eval()
        
        # 메트릭 초기화
        test_metrics = RegressionMetrics(self.device)
        
        print("🧪 모델 테스트 중...")
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.data_loader.test_loader, desc="Testing"):
                # 데이터를 디바이스로 이동
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 순전파
                predictions = self.model(inputs)
                
                # 손실 계산
                loss = self.loss_function(predictions, targets)
                
                # 메트릭 업데이트
                test_metrics.update(predictions, targets, loss)
        
        # 테스트 메트릭 계산 및 출력
        metrics = test_metrics.compute()
        
        print("🧪 테스트 결과:")
        test_metrics.print_metrics()
        
        return metrics
    
    def save_checkpoint(self, 
                       metrics: Dict[str, Any], 
                       is_best: bool = False,
                       checkpoint_type: str = "epoch"):
        """
        체크포인트 저장
        
        Args:
            metrics (Dict): 현재 메트릭
            is_best (bool): 최고 성능 모델 여부
            checkpoint_type (str): 체크포인트 타입
        """
        # 저장할 정보 구성
        checkpoint_info = {
            'train_metrics': metrics.get('train', {}),
            'val_metrics': metrics.get('val', {}),
            'best_val_mae': self.best_val_mae,
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None
        }
        
        # 모델 저장
        if is_best:
            save_model_checkpoint(
                self.model, 
                str(self.best_model_path),
                epoch=self.current_epoch,
                loss=metrics.get('val', {}).get('avg_loss'),
                optimizer_state=self.optimizer.state_dict(),
                **checkpoint_info
            )
        
        # 마지막 모델 저장
        save_model_checkpoint(
            self.model,
            str(self.last_model_path),
            epoch=self.current_epoch,
            loss=metrics.get('val', {}).get('avg_loss'),
            optimizer_state=self.optimizer.state_dict(),
            **checkpoint_info
        )
    
    def train(self, 
             num_epochs: int,
             save_every: int = 5,
             validate_every: int = 1,
             early_stopping: bool = True,
             verbose: bool = True) -> Dict[str, Any]:
        """
        모델 훈련
        
        Args:
            num_epochs (int): 훈련 에포크 수
            save_every (int): 체크포인트 저장 주기
            validate_every (int): 검증 수행 주기
            early_stopping (bool): 조기 종료 사용 여부
            verbose (bool): 상세 출력 여부
            
        Returns:
            Dict[str, Any]: 훈련 결과
        """
        print(f"🍉 수박 당도 예측 모델 훈련 시작")
        print(f"   📊 에포크: {num_epochs}")
        print(f"   🔄 검증 주기: {validate_every}")
        print(f"   💾 저장 주기: {save_every}")
        print(f"   ⏹️ 조기 종료: {early_stopping}")
        print("=" * 60)
        
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 훈련
            train_metrics = self.train_epoch()
            
            # 검증
            if epoch % validate_every == 0:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {}
            
            # 학습률 스케줄링
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics.get('mae', train_metrics['mae']))
                else:
                    self.scheduler.step()
            
            # 메트릭 추적
            if val_metrics:
                self.metrics_tracker.add_epoch_metrics(
                    train_metrics, 
                    val_metrics,
                    train_metrics['avg_loss']
                )
            
            # 최고 성능 모델 체크
            is_best = False
            if val_metrics and val_metrics.get('mae', float('inf')) < self.best_val_mae:
                self.best_val_mae = val_metrics['mae']
                is_best = True
                self.early_stopping_counter = 0
            elif val_metrics:
                self.early_stopping_counter += 1
            
            # 체크포인트 저장
            if epoch % save_every == 0 or is_best or epoch == num_epochs - 1:
                metrics_dict = {'train': train_metrics, 'val': val_metrics}
                self.save_checkpoint(metrics_dict, is_best)
            
            # 진행 상황 출력
            if verbose:
                epoch_time = time.time() - epoch_start_time
                print(f"\n📊 Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"   🚂 Train - MAE: {train_metrics['mae']:.3f}, "
                      f"RMSE: {train_metrics['rmse']:.3f}, "
                      f"R²: {train_metrics['r2_score']:.3f}")
                
                if val_metrics:
                    print(f"   🔍 Val   - MAE: {val_metrics['mae']:.3f}, "
                          f"RMSE: {val_metrics['rmse']:.3f}, "
                          f"R²: {val_metrics['r2_score']:.3f}")
                    print(f"   {'🏆 BEST!' if is_best else ''}")
                
                # 학습률 출력
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"   🎚️ LR: {current_lr:.2e}")
            
            # 조기 종료 체크
            if early_stopping and self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\n⏹️ 조기 종료: {self.early_stopping_patience} 에포크 동안 개선 없음")
                break
        
        # 훈련 완료
        total_training_time = time.time() - training_start_time
        print(f"\n✅ 훈련 완료!")
        print(f"   ⏱️ 총 시간: {total_training_time/3600:.2f}시간")
        print(f"   🏆 최고 Val MAE: {self.best_val_mae:.3f}")
        
        # 결과 저장
        self._save_training_results()
        
        # 최종 테스트
        final_test_metrics = self.test_model()
        
        return {
            'best_val_mae': self.best_val_mae,
            'total_epochs': self.current_epoch + 1,
            'training_time': total_training_time,
            'final_test_metrics': final_test_metrics,
            'metrics_tracker': self.metrics_tracker
        }
    
    def _save_training_results(self):
        """훈련 결과 저장"""
        # 메트릭 요약 저장
        metrics_path = self.save_dir / "metrics_summary.json"
        self.metrics_tracker.save_metrics_summary(str(metrics_path))
        
        # 훈련 곡선 저장
        curves_path = self.save_dir / "training_curves.png"
        self.metrics_tracker.plot_training_curves(str(curves_path), show=False)
        
        print(f"📊 훈련 결과 저장됨: {self.save_dir}")


def create_trainer_from_config(config_path: str, 
                              data_path: str,
                              experiment_name: Optional[str] = None) -> WatermelonTrainer:
    """
    설정 파일로부터 트레이너 생성
    
    Args:
        config_path (str): 훈련 설정 파일 경로
        data_path (str): 데이터 경로
        experiment_name (str, optional): 실험 이름
        
    Returns:
        WatermelonTrainer: 생성된 트레이너
    """
    # 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 디바이스: {device}")
    
    # 모델 생성
    from models.vgg_watermelon import create_vgg_watermelon
    model_config = config.get('model', {})
    model = create_vgg_watermelon(**model_config)
    model.print_model_info()
    
    # 데이터 로더 생성
    from training.data_loader import create_data_loaders
    data_config = config.get('data', {})
    
    # create_data_loaders가 지원하는 파라미터만 필터링
    supported_params = {
        'train_ratio', 'val_ratio', 'test_ratio', 'batch_size', 'num_workers', 
        'pin_memory', 'use_augmentation', 'stratify_by_sweetness', 'random_seed', 
        'split_file', 'config_path'
    }
    filtered_data_config = {k: v for k, v in data_config.items() if k in supported_params}
    
    data_loader = create_data_loaders(data_path, **filtered_data_config)
    
    # 손실 함수 생성
    loss_config = config.get('loss', {'type': 'mse'})
    loss_function = create_loss_function(loss_config)
    
    # 옵티마이저 생성
    optimizer_config = config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'adam').lower()
    lr = float(optimizer_config.get('lr', 0.001))
    weight_decay = float(optimizer_config.get('weight_decay', 1e-4))
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_type}")
    
    # 스케줄러 생성
    scheduler = None
    scheduler_config = config.get('scheduler', {})
    if scheduler_config.get('enabled', False):
        scheduler_type = scheduler_config.get('type', 'plateau').lower()
        
        if scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5)
            )
        elif scheduler_type == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', 50),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
    
    # 실험 이름 설정
    if experiment_name is None:
        experiment_name = f"watermelon_{model_config.get('name', 'vgg')}"
    
    # 트레이너 생성
    trainer = WatermelonTrainer(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        experiment_name=experiment_name
    )
    
    return trainer


if __name__ == "__main__":
    # 트레이너 테스트
    print("🧪 트레이너 모듈 테스트")
    
    # 기본 설정으로 간단한 테스트
    try:
        # 설정 파일이 있다면 사용
        config_path = "../../configs/training.yaml"
        data_path = "../../watermelon_sound_data"
        
        if Path(config_path).exists():
            trainer = create_trainer_from_config(config_path, data_path, "test_run")
            
            # 짧은 훈련 실행
            results = trainer.train(num_epochs=2, verbose=True)
            
            print("✅ 트레이너 테스트 완료")
            print(f"   🏆 최고 Val MAE: {results['best_val_mae']:.3f}")
        else:
            print(f"⚠️ 설정 파일 없음: {config_path}")
            
    except Exception as e:
        print(f"❌ 트레이너 테스트 실패: {e}")
        import traceback
        traceback.print_exc() 