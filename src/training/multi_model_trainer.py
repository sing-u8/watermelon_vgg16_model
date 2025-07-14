"""
Multi-Model Trainer Module
다중 모델 아키텍처 지원 통합 트레이너
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from typing import Dict, Any, Optional, List, Tuple, Union
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
from models.efficientnet_watermelon import EfficientNetWatermelon, save_efficientnet_checkpoint
from models.melspec_cnn_watermelon import MelSpecCNNWatermelon, save_melspec_cnn_checkpoint
from training.data_loader import WatermelonDataLoader
from training.loss_functions import create_loss_function
from training.metrics import RegressionMetrics, MetricsTracker

# 지원 모델 타입
ModelType = Union[VGGWatermelon, EfficientNetWatermelon, MelSpecCNNWatermelon]


class MultiModelTrainer:
    """
    다중 모델 아키텍처 지원 통합 트레이너 클래스
    """
    
    def __init__(self, 
                 model: ModelType,
                 data_loader: WatermelonDataLoader,
                 loss_function: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 save_dir: str = "experiments",
                 experiment_name: str = "multi_model_training",
                 model_type: str = "vgg"):
        """
        MultiModelTrainer 초기화
        
        Args:
            model (ModelType): 훈련할 모델
            data_loader (WatermelonDataLoader): 데이터 로더
            loss_function (nn.Module): 손실 함수
            optimizer (optim.Optimizer): 옵티마이저
            device (torch.device): 학습 디바이스
            scheduler (optim.lr_scheduler._LRScheduler, optional): 학습률 스케줄러
            save_dir (str): 저장 디렉토리
            experiment_name (str): 실험 이름
            model_type (str): 모델 타입 ("vgg", "efficientnet", "melspec_cnn")
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.model_type = model_type
        
        # 실험 디렉토리 생성
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 메트릭 추적기 초기화
        self.metrics_tracker = MetricsTracker()
        
        # 최적 모델 상태 추적
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        self.best_epoch = 0
        
        # 시간 측정
        self.start_time = None
        self.total_training_time = 0
        
        print(f"🚀 MultiModelTrainer 초기화 완료")
        print(f"   📊 모델 타입: {model_type}")
        print(f"   🎯 실험 이름: {experiment_name}")
        print(f"   📁 저장 경로: {self.experiment_dir}")
        print(f"   🔧 디바이스: {device}")
        
    def train_epoch(self) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.model.train()
        train_loader = self.data_loader.get_train_loader()
        
        total_loss = 0.0
        predictions = []
        targets = []
        
        # 진행 바 설정
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 그래디언트 초기화
            self.optimizer.zero_grad()
            
            # 순전파
            output = self.model(data)
            loss = self.loss_function(output, target)
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            # 메트릭 수집
            total_loss += loss.item()
            predictions.extend(output.detach().cpu().numpy())
            targets.extend(target.detach().cpu().numpy())
            
            # 진행 바 업데이트
            pbar.set_postfix({'loss': loss.item():.4f})
        
        # 에포크 메트릭 계산
        avg_loss = total_loss / len(train_loader)
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        metrics = RegressionMetrics.calculate_metrics(predictions, targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """한 에포크 검증"""
        self.model.eval()
        val_loader = self.data_loader.get_val_loader()
        
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating", leave=False)
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # 순전파
                output = self.model(data)
                loss = self.loss_function(output, target)
                
                # 메트릭 수집
                total_loss += loss.item()
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
                
                # 진행 바 업데이트
                pbar.set_postfix({'loss': loss.item():.4f})
        
        # 에포크 메트릭 계산
        avg_loss = total_loss / len(val_loader)
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        metrics = RegressionMetrics.calculate_metrics(predictions, targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def test_model(self) -> Dict[str, float]:
        """모델 테스트"""
        self.model.eval()
        test_loader = self.data_loader.get_test_loader()
        
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing", leave=False)
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # 순전파
                output = self.model(data)
                loss = self.loss_function(output, target)
                
                # 메트릭 수집
                total_loss += loss.item()
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
                
                # 진행 바 업데이트
                pbar.set_postfix({'loss': loss.item():.4f})
        
        # 테스트 메트릭 계산
        avg_loss = total_loss / len(test_loader)
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        metrics = RegressionMetrics.calculate_metrics(predictions, targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, 
                       metrics: Dict[str, Any], 
                       epoch: int,
                       is_best: bool = False,
                       checkpoint_type: str = "epoch"):
        """체크포인트 저장 (모델 타입별)"""
        checkpoint_name = f"{checkpoint_type}_checkpoint"
        if is_best:
            checkpoint_name = "best_model"
        
        save_path = self.experiment_dir / f"{checkpoint_name}.pth"
        
        # 모델 타입에 따라 적절한 저장 함수 사용
        if self.model_type == "vgg":
            save_model_checkpoint(
                self.model,
                str(save_path),
                epoch=epoch,
                loss=metrics.get('loss', 0.0),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
                metrics=metrics
            )
        elif self.model_type == "efficientnet":
            save_efficientnet_checkpoint(
                self.model,
                str(save_path),
                epoch=epoch,
                loss=metrics.get('loss', 0.0),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
                metrics=metrics
            )
        elif self.model_type == "melspec_cnn":
            save_melspec_cnn_checkpoint(
                self.model,
                str(save_path),
                epoch=epoch,
                loss=metrics.get('loss', 0.0),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
                metrics=metrics
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
    
    def train(self, 
             num_epochs: int,
             save_every: int = 5,
             validate_every: int = 1,
             early_stopping: bool = True,
             patience: int = 5,
             verbose: bool = True) -> Dict[str, Any]:
        """모델 훈련"""
        
        print(f"🚀 {self.model_type.upper()} 모델 훈련 시작!")
        print(f"   📊 에포크: {num_epochs}")
        print(f"   🎯 Early Stopping: {early_stopping} (patience: {patience})")
        print(f"   💾 체크포인트 저장 주기: {save_every}")
        
        self.start_time = time.time()
        
        # Early stopping 변수
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # 훈련
            train_metrics = self.train_epoch()
            
            # 검증
            val_metrics = None
            if epoch % validate_every == 0:
                val_metrics = self.validate_epoch()
                
                # 학습률 스케줄러 업데이트
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
            
            # 메트릭 추적
            self.metrics_tracker.add_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # 최적 모델 체크
            is_best = False
            if val_metrics and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_mae = val_metrics['mae']
                self.best_epoch = epoch
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 체크포인트 저장
            if epoch % save_every == 0 or is_best:
                current_metrics = val_metrics if val_metrics else train_metrics
                self.save_checkpoint(current_metrics, epoch, is_best)
            
            # 진행 상황 출력
            if verbose:
                epoch_time = time.time() - epoch_start_time
                print(f"📊 Epoch {epoch:3d}/{num_epochs}")
                print(f"   🏃 Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}")
                if val_metrics:
                    print(f"   ✅ Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}")
                print(f"   ⏱️ Time: {epoch_time:.2f}s")
                if self.scheduler:
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(f"   📈 LR: {current_lr:.2e}")
                print()
            
            # Early stopping 체크
            if early_stopping and patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch} (patience: {patience})")
                break
        
        self.total_training_time = time.time() - self.start_time
        
        # 최종 테스트
        print("🧪 최종 테스트 실행...")
        test_metrics = self.test_model()
        
        # 훈련 결과 저장
        training_results = {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'test_metrics': test_metrics,
            'total_training_time': self.total_training_time,
            'model_type': self.model_type,
            'experiment_name': self.experiment_name
        }
        
        self._save_training_results(training_results)
        
        # 성공 파일 생성
        success_file = self.experiment_dir / "SUCCESS"
        success_file.write_text(f"Training completed successfully!\n"
                               f"Best epoch: {self.best_epoch}\n"
                               f"Best val MAE: {self.best_val_mae:.4f}\n"
                               f"Test MAE: {test_metrics['mae']:.4f}\n"
                               f"Model type: {self.model_type}\n"
                               f"Total time: {self.total_training_time:.2f}s")
        
        print(f"🎉 {self.model_type.upper()} 모델 훈련 완료!")
        print(f"   🏆 최고 성능 (Epoch {self.best_epoch}): Val MAE {self.best_val_mae:.4f}")
        print(f"   🧪 테스트 성능: MAE {test_metrics['mae']:.4f}")
        print(f"   ⏱️ 총 훈련 시간: {self.total_training_time:.2f}초")
        
        return training_results
    
    def _save_training_results(self, results: Dict[str, Any]):
        """훈련 결과 저장"""
        results_file = self.experiment_dir / "training_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        print(f"💾 훈련 결과 저장 완료: {results_file}")


def create_model_from_config(config: Dict[str, Any], model_type: str) -> ModelType:
    """설정에서 모델 생성"""
    model_config = config.get('model', {})
    
    if model_type == "vgg":
        from models.vgg_watermelon import create_vgg_watermelon
        return create_vgg_watermelon(**model_config)
    elif model_type == "efficientnet":
        from models.efficientnet_watermelon import create_efficientnet_watermelon
        return create_efficientnet_watermelon(**model_config)
    elif model_type == "melspec_cnn":
        from models.melspec_cnn_watermelon import create_melspec_cnn_watermelon
        return create_melspec_cnn_watermelon(**model_config)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")


def create_multi_model_trainer_from_config(config_path: str, 
                                         data_path: str,
                                         model_type: str = "vgg",
                                         experiment_name: Optional[str] = None) -> MultiModelTrainer:
    """
    설정 파일에서 MultiModelTrainer 생성
    
    Args:
        config_path (str): 설정 파일 경로
        data_path (str): 데이터 경로
        model_type (str): 모델 타입 ("vgg", "efficientnet", "melspec_cnn")
        experiment_name (str, optional): 실험 이름
        
    Returns:
        MultiModelTrainer: 생성된 트레이너
    """
    # 설정 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 실험 이름 설정
    if experiment_name is None:
        experiment_name = config.get('experiment_name', f'{model_type}_experiment')
    
    # 디바이스 설정
    device_config = config.get('device', 'auto')
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    # 모델 생성
    model = create_model_from_config(config, model_type)
    
    # 데이터 로더 생성
    data_loader = WatermelonDataLoader(
        data_root=data_path,
        **config.get('data', {})
    )
    
    # 손실 함수 생성
    loss_config = config.get('training', {})
    loss_function = create_loss_function(
        loss_config.get('loss_function', 'mse'),
        huber_delta=loss_config.get('huber_delta', 1.0)
    )
    
    # 옵티마이저 생성
    optimizer_name = loss_config.get('optimizer', 'adam')
    learning_rate = loss_config.get('learning_rate', 0.001)
    weight_decay = loss_config.get('weight_decay', 1e-4)
    
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_name}")
    
    # 스케줄러 생성
    scheduler = None
    scheduler_config = loss_config.get('scheduler', 'step')
    if scheduler_config == 'step':
        step_size = loss_config.get('step_size', 7)
        gamma = loss_config.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_config == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # 트레이너 생성
    trainer = MultiModelTrainer(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        save_dir=config.get('logging', {}).get('save_dir', 'experiments'),
        experiment_name=experiment_name,
        model_type=model_type
    )
    
    return trainer 