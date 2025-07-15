#!/usr/bin/env python3
"""
Real Architecture Training Script
실제 EfficientNet과 MelSpecCNN 아키텍처로 훈련하는 스크립트
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 모델 및 유틸리티 import
from src.models.efficientnet_watermelon import create_efficientnet_watermelon
from src.models.melspec_cnn_watermelon import create_melspec_cnn_watermelon
from src.data.watermelon_dataset import WatermelonDataset
from src.training.data_loader import create_data_loaders_stratified
from src.training.loss_functions import get_loss_function
from src.training.metrics import calculate_metrics

warnings.filterwarnings('ignore')


class RealArchitectureTrainer:
    """실제 아키텍처를 사용하는 훈련 클래스"""
    
    def __init__(self, model_type: str, config: dict, save_dir: str):
        self.model_type = model_type
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ 디바이스: {self.device}")
        
        # 모델 생성
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 손실 함수 및 옵티마이저 설정
        self.criterion = get_loss_function(
            config['training']['loss_function'],
            config['training'].get('huber_delta', 1.0)
        )
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.best_val_mae = float('inf')
        self.best_epoch = 0
        
    def _create_model(self) -> nn.Module:
        """모델 생성"""
        model_config = self.config['model']
        
        if self.model_type == 'efficientnet':
            model = create_efficientnet_watermelon(
                pretrained=model_config.get('pretrained', False),
                dropout_rate=model_config.get('dropout_rate', 0.7),
                num_fc_layers=model_config.get('num_fc_layers', 2),
                fc_hidden_size=model_config.get('fc_hidden_size', 256)
            )
            print("✅ EfficientNet 모델 생성 완료")
            
        elif self.model_type == 'melspec_cnn':
            model = create_melspec_cnn_watermelon(
                input_channels=model_config.get('input_channels', 3),
                base_channels=model_config.get('base_channels', 32),
                dropout_rate=model_config.get('dropout_rate', 0.7)
            )
            print("✅ MelSpecCNN 모델 생성 완료")
            
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        # 모델 정보 출력
        model.print_model_info()
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """옵티마이저 생성"""
        optimizer_name = self.config['training'].get('optimizer', 'adam').lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training'].get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config['training'].get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_name}")
    
    def _create_scheduler(self):
        """스케줄러 생성"""
        scheduler_type = self.config['training'].get('scheduler', 'step')
        
        if scheduler_type == 'step':
            step_size = self.config['training'].get('step_size', 7)
            gamma = self.config['training'].get('gamma', 0.7)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            targets = targets.float().unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # 메트릭 계산
            total_loss += loss.item()
            mae = torch.mean(torch.abs(outputs - targets)).item()
            total_mae += mae
            num_batches += 1
            
            # Progress bar 업데이트
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{mae:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def validate_epoch(self, val_loader: DataLoader) -> tuple:
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                targets = targets.float().unsqueeze(1)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                mae = torch.mean(torch.abs(outputs - targets)).item()
                total_mae += mae
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> dict:
        """모델 훈련"""
        print(f"🚀 {self.model_type.upper()} 모델 훈련 시작!")
        print(f"   📊 에포크: {num_epochs}")
        print(f"   🖥️ 디바이스: {self.device}")
        print("=" * 60)
        
        early_stopping_patience = self.config['training']['early_stopping'].get('patience', 10)
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            start_time = datetime.now()
            
            # 훈련
            train_loss, train_mae = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate_epoch(val_loader)
            
            # 스케줄러 업데이트
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 기록 저장
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_maes.append(train_mae)
            self.val_maes.append(val_mae)
            
            # 최고 성능 업데이트
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_epoch = epoch + 1
                early_stopping_counter = 0
                
                # 최고 모델 저장
                self._save_checkpoint(epoch + 1, is_best=True)
                print("   🏆 BEST!")
            else:
                early_stopping_counter += 1
            
            # 훈련 시간 계산
            epoch_time = (datetime.now() - start_time).total_seconds()
            
            # 현재 학습률
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 에포크 결과 출력
            print(f"📊 Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"   🚂 Train - Loss: {train_loss:.3f}, MAE: {train_mae:.3f}")
            print(f"   🔍 Val   - Loss: {val_loss:.3f}, MAE: {val_mae:.3f}")
            print(f"   🎚️ LR: {current_lr:.2e}")
            
            # 조기 종료 확인
            if early_stopping_counter >= early_stopping_patience:
                print(f"\n⏹️ 조기 종료: {early_stopping_patience} 에포크 동안 개선 없음")
                break
        
        # 훈련 완료
        print("\n✅ 훈련 완료!")
        print(f"   🏆 최고 Val MAE: {self.best_val_mae:.4f} (에포크 {self.best_epoch})")
        
        # 훈련 결과 저장
        self._save_training_results()
        self._plot_training_curves()
        
        return {
            'best_val_mae': self.best_val_mae,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_maes': self.train_maes,
            'val_maes': self.val_maes
        }
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_mae': self.best_val_mae,
            'model_type': self.model_type,
            'model_config': self.config['model']
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if is_best:
            checkpoint_path = self.save_dir / "best_model.pth"
        else:
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, checkpoint_path)
    
    def _save_training_results(self):
        """훈련 결과 저장"""
        results = {
            'model_type': self.model_type,
            'best_val_mae': self.best_val_mae,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.train_losses),
            'final_train_mae': self.train_maes[-1],
            'final_val_mae': self.val_maes[-1],
            'config': self.config
        }
        
        with open(self.save_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    def _plot_training_curves(self):
        """훈련 곡선 시각화"""
        plt.figure(figsize=(12, 4))
        
        # Loss 곡선
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Val Loss', color='red')
        plt.title(f'{self.model_type.upper()} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # MAE 곡선
        plt.subplot(1, 2, 2)
        plt.plot(self.train_maes, label='Train MAE', color='blue')
        plt.plot(self.val_maes, label='Val MAE', color='red')
        plt.axhline(y=self.best_val_mae, color='red', linestyle='--', 
                   label=f'Best Val MAE: {self.best_val_mae:.4f}')
        plt.title(f'{self.model_type.upper()} - Training MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_config(model_type: str) -> dict:
    """모델 타입별 설정 생성"""
    base_config = {
        'data': {
            'data_root': 'watermelon_sound_data',
            'train_ratio': 0.70,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True,
            'audio_params': {
                'sample_rate': 16000,
                'n_mels': 128,
                'n_fft': 2048,
                'hop_length': 512,
                'win_length': 2048,
                'f_min': 0.0,
                'f_max': 11025.0
            },
            'augmentation': {
                'enabled': True,
                'noise_factor': 0.005,
                'time_shift_max': 0.1,
                'pitch_shift_max': 2.0,
                'volume_change_max': 0.1
            },
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        },
        'training': {
            'epochs': 30,
            'learning_rate': 0.0001,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'scheduler': 'step',
            'step_size': 10,
            'gamma': 0.7,
            'loss_function': 'mse',
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 0.001
            }
        }
    }
    
    if model_type == 'efficientnet':
        base_config['model'] = {
            'pretrained': False,  # 처음부터 훈련
            'dropout_rate': 0.7,
            'num_fc_layers': 2,
            'fc_hidden_size': 256
        }
    elif model_type == 'melspec_cnn':
        base_config['model'] = {
            'input_channels': 3,
            'base_channels': 32,
            'dropout_rate': 0.7
        }
    
    return base_config


def train_model(model_type: str, data_root: str) -> dict:
    """단일 모델 훈련"""
    # 설정 생성
    config = create_config(model_type)
    
    # 저장 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/real_{model_type}_{timestamp}"
    
    # 데이터 로더 생성
    print(f"📂 데이터 로더 생성 중... ({model_type})")
    train_loader, val_loader, test_loader = create_data_loaders_stratified(
        data_root=data_root,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        audio_params=config['data']['audio_params'],
        augmentation_params=config['data']['augmentation'],
        normalization_params=config['data']['normalization']
    )
    
    # 트레이너 생성 및 훈련
    trainer = RealArchitectureTrainer(model_type, config, save_dir)
    results = trainer.train(train_loader, val_loader, config['training']['epochs'])
    
    # 설정 저장
    with open(Path(save_dir) / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Real Architecture Training Script")
    parser.add_argument("--model", choices=["efficientnet", "melspec_cnn", "both"], 
                       default="both", help="훈련할 모델")
    parser.add_argument("--data-root", default="watermelon_sound_data", 
                       help="데이터 루트 디렉토리")
    
    args = parser.parse_args()
    
    print("🍉 Real Architecture Training Script!")
    print(f"   🎯 훈련 모델: {args.model}")
    print(f"   📂 데이터 경로: {args.data_root}")
    print()
    
    results = {}
    
    if args.model in ["efficientnet", "both"]:
        print("🚀 EfficientNet 모델 훈련 시작!")
        results['efficientnet'] = train_model('efficientnet', args.data_root)
        print("✅ EfficientNet 훈련 완료!\n")
    
    if args.model in ["melspec_cnn", "both"]:
        print("🚀 MelSpecCNN 모델 훈련 시작!")
        results['melspec_cnn'] = train_model('melspec_cnn', args.data_root)
        print("✅ MelSpecCNN 훈련 완료!\n")
    
    # 최종 결과 출력
    print("🎉 모든 훈련 완료!")
    for model_name, result in results.items():
        print(f"   {model_name.upper()}: Best Val MAE = {result['best_val_mae']:.4f}")


if __name__ == "__main__":
    main() 