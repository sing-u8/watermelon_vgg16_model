#!/usr/bin/env python3
"""
Direct Model Training Script
EfficientNet과 MelSpecCNN을 직접 훈련하는 스크립트 (VGG 우회)

# EfficientNet vs MelSpecCNN 성능 비교 (권장)
python scripts/train_direct_models.py --model compare

# 개별 모델 훈련
python scripts/train_direct_models.py --model efficientnet
python scripts/train_direct_models.py --model melspec_cnn
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.efficientnet_watermelon import create_efficientnet_watermelon
from src.models.melspec_cnn_watermelon import create_melspec_cnn_watermelon
from src.training.data_loader import create_data_loaders


class DirectModelTrainer:
    """직접 모델 훈련 클래스"""
    
    def __init__(self, model, data_loader, save_dir, device='cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.save_dir = Path(save_dir)
        
        # 저장 디렉토리 생성
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 손실 함수 및 옵티마이저 설정
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0001,
            weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.7
        )
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        
    def train_epoch(self, epoch):
        """한 에폭 훈련"""
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.data_loader.train_loader,
            desc=f"Epoch {epoch+1} 훈련",
            leave=False
        )
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # 진행률 업데이트
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{running_loss/num_batches:.4f}'
            })
        
        avg_train_loss = running_loss / num_batches
        self.train_losses.append(avg_train_loss)
        
        return avg_train_loss
    
    def validate_epoch(self, epoch):
        """한 에폭 검증"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.data_loader.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                
                running_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy())
        
        avg_val_loss = running_loss / len(self.data_loader.val_loader)
        self.val_losses.append(avg_val_loss)
        
        # 메트릭 계산
        val_mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_targets)))
        self.val_maes.append(val_mae)
        
        # 최고 성능 업데이트
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_checkpoint(epoch, 'best_loss')
        
        if val_mae < self.best_val_mae:
            self.best_val_mae = val_mae
            self.save_checkpoint(epoch, 'best_mae')
        
        return avg_val_loss, val_mae
    
    def save_checkpoint(self, epoch, name):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maes': self.val_maes
        }
        
        save_path = self.save_dir / f"{name}_epoch_{epoch+1}.pth"
        torch.save(checkpoint, save_path)
        print(f"   💾 체크포인트 저장: {save_path}")
    
    def train(self, num_epochs=30):
        """전체 훈련 과정"""
        print(f"🚀 훈련 시작! (총 {num_epochs} 에폭)")
        print(f"   🖥️ 디바이스: {self.device}")
        
        for epoch in range(num_epochs):
            # 훈련
            train_loss = self.train_epoch(epoch)
            
            # 검증
            val_loss, val_mae = self.validate_epoch(epoch)
            
            # 학습률 조정
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 결과 출력
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val MAE: {val_mae:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # Early stopping (간단한 버전)
            if epoch > 10 and val_loss > self.best_val_loss * 1.5:
                print("   ⏹️ Early stopping triggered")
                break
        
        print(f"✅ 훈련 완료!")
        print(f"   🏆 최고 Val Loss: {self.best_val_loss:.4f}")
        print(f"   🏆 최고 Val MAE: {self.best_val_mae:.4f}")
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maes': self.val_maes
        }


def train_efficientnet(data_path="watermelon_sound_data"):
    """EfficientNet 모델 훈련"""
    print("🚀 EfficientNet 모델 직접 훈련 시작!")
    
    # 모델 생성
    print("🔧 EfficientNet 모델 생성 중...")
    model = create_efficientnet_watermelon(
        pretrained=False,
        dropout_rate=0.7,
        num_fc_layers=2,
        fc_hidden_size=256
    )
    model.print_model_info()
    
    # 데이터 로더 생성
    print("📂 데이터 로더 생성 중...")
    data_loader = create_data_loaders(
        data_path=data_path,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        use_augmentation=True,
        stratify_by_sweetness=True,
        random_seed=42
    )
    
    # 저장 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/direct_efficientnet_{timestamp}"
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 트레이너 생성 및 훈련
    trainer = DirectModelTrainer(model, data_loader, save_dir, str(device))
    results = trainer.train(num_epochs=30)
    
    return results, save_dir


def train_melspec_cnn(data_path="watermelon_sound_data"):
    """MelSpecCNN 모델 훈련"""
    print("🚀 MelSpecCNN 모델 직접 훈련 시작!")
    
    # 모델 생성
    print("🔧 MelSpecCNN 모델 생성 중...")
    model = create_melspec_cnn_watermelon(
        input_channels=3,
        base_channels=32,
        dropout_rate=0.7
    )
    model.print_model_info()
    
    # 데이터 로더 생성
    print("📂 데이터 로더 생성 중...")
    data_loader = create_data_loaders(
        data_path=data_path,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        use_augmentation=True,
        stratify_by_sweetness=True,
        random_seed=42
    )
    
    # 저장 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/direct_melspec_cnn_{timestamp}"
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 트레이너 생성 및 훈련
    trainer = DirectModelTrainer(model, data_loader, save_dir, str(device))
    results = trainer.train(num_epochs=30)
    
    return results, save_dir


def compare_models(data_path="watermelon_sound_data"):
    """두 모델을 훈련하고 성능 비교"""
    print("🍉 EfficientNet vs MelSpecCNN 성능 비교 시작!")
    print("="*60)
    
    # EfficientNet 훈련
    print("\n1️⃣ EfficientNet 훈련")
    print("-"*40)
    efficientnet_results, efficientnet_dir = train_efficientnet(data_path)
    
    print("\n2️⃣ MelSpecCNN 훈련")
    print("-"*40)
    melspec_results, melspec_dir = train_melspec_cnn(data_path)
    
    # 결과 비교
    print("\n📊 성능 비교 결과")
    print("="*60)
    print(f"🔸 EfficientNet:")
    print(f"   최고 Val Loss: {efficientnet_results['best_val_loss']:.4f}")
    print(f"   최고 Val MAE:  {efficientnet_results['best_val_mae']:.4f}")
    print(f"   저장 경로: {efficientnet_dir}")
    
    print(f"\n🔸 MelSpecCNN:")
    print(f"   최고 Val Loss: {melspec_results['best_val_loss']:.4f}")
    print(f"   최고 Val MAE:  {melspec_results['best_val_mae']:.4f}")
    print(f"   저장 경로: {melspec_dir}")
    
    # 승자 결정
    if efficientnet_results['best_val_mae'] < melspec_results['best_val_mae']:
        print(f"\n🏆 승자: EfficientNet (MAE 차이: {melspec_results['best_val_mae'] - efficientnet_results['best_val_mae']:.4f})")
    else:
        print(f"\n🏆 승자: MelSpecCNN (MAE 차이: {efficientnet_results['best_val_mae'] - melspec_results['best_val_mae']:.4f})")
    
    return efficientnet_results, melspec_results


def main():
    parser = argparse.ArgumentParser(description="Direct Model Training")
    parser.add_argument('--model', 
                       choices=['efficientnet', 'melspec_cnn', 'compare'], 
                       default='compare',
                       help='훈련할 모델')
    parser.add_argument('--data', 
                       default='watermelon_sound_data',
                       help='데이터 경로')
    
    args = parser.parse_args()
    
    if args.model == 'efficientnet':
        train_efficientnet(args.data)
    elif args.model == 'melspec_cnn':
        train_melspec_cnn(args.data)
    elif args.model == 'compare':
        compare_models(args.data)


if __name__ == "__main__":
    main() 