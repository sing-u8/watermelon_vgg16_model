#!/usr/bin/env python3
"""
Real Architecture Training Script (Simplified)
기존 인프라를 활용하여 실제 EfficientNet과 MelSpecCNN 훈련
"""

import os
import sys
import argparse
import yaml
import torch
import warnings
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 기존 트레이너 import
from src.training.trainer import WatermelonTrainer
from src.training.data_loader import create_data_loaders
from src.models.efficientnet_watermelon import create_efficientnet_watermelon
from src.models.melspec_cnn_watermelon import create_melspec_cnn_watermelon

warnings.filterwarnings('ignore')


class RealModelTrainer(WatermelonTrainer):
    """실제 모델을 사용하는 트레이너 (기존 인프라 상속)"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, save_dir, device='auto'):
        # 기존 트레이너 초기화하지만 모델은 직접 설정
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🖥️ 디바이스: {self.device}")
        
        # 모델을 디바이스로 이동
        self.model.to(self.device)
        
        # 모델 정보 출력
        self.model.print_model_info()
        
        # 훈련 기록 초기화
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_mae = float('inf')
        self.best_epoch = 0


def create_real_model(model_type: str, config: dict):
    """실제 모델 생성"""
    model_config = config.get('model', {})
    
    if model_type == 'efficientnet':
        model = create_efficientnet_watermelon(
            pretrained=model_config.get('pretrained', False),
            dropout_rate=model_config.get('dropout_rate', 0.7),
            num_fc_layers=model_config.get('num_fc_layers', 2),
            fc_hidden_size=model_config.get('fc_hidden_size', 256)
        )
        print("✅ EfficientNet 모델 생성 완료")
        
    elif model_type == 'melspec_cnn':
        model = create_melspec_cnn_watermelon(
            input_channels=model_config.get('input_channels', 3),
            base_channels=model_config.get('base_channels', 32),
            dropout_rate=model_config.get('dropout_rate', 0.7)
        )
        print("✅ MelSpecCNN 모델 생성 완료")
        
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    return model


def create_config(model_type: str) -> dict:
    """모델별 기본 설정 생성"""
    base_config = {
        "experiment_name": f"real_{model_type}_exp",
        "data": {
            "data_root": "watermelon_sound_data",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "batch_size": 16,
            "num_workers": 4,
            "pin_memory": True,
            "audio_params": {
                "sample_rate": 22050,
                "n_mels": 128,
                "n_fft": 2048,
                "hop_length": 512,
                "win_length": 2048,
                "f_min": 0.0,
                "f_max": 11025.0
            },
            "augmentation": {
                "enabled": True,
                "noise_factor": 0.005,
                "time_shift_max": 0.1,
                "pitch_shift_max": 2.0,
                "volume_change_max": 0.1
            },
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        },
        "training": {
            "epochs": 30,
            "learning_rate": 0.0001,
            "weight_decay": 1e-4,
            "optimizer": "adam",
            "scheduler": "step",
            "step_size": 10,
            "gamma": 0.7,
            "loss_function": "mse",
            "huber_delta": 1.0,
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "min_delta": 0.001,
                "monitor": "val_loss",
                "mode": "min"
            }
        },
        "logging": {
            "save_dir": f"experiments/real_{model_type}_exp",
            "use_tensorboard": True
        },
        "device": "auto",
        "seed": 42
    }
    
    # 모델별 특별 설정
    if model_type == 'efficientnet':
        base_config['model'] = {
            "pretrained": False,
            "dropout_rate": 0.7,
            "num_fc_layers": 2,
            "fc_hidden_size": 256
        }
    elif model_type == 'melspec_cnn':
        base_config['model'] = {
            "input_channels": 3,
            "base_channels": 32,
            "dropout_rate": 0.7
        }
    
    return base_config


def train_real_model(model_type: str, data_path: str) -> dict:
    """실제 모델 훈련"""
    print(f"🚀 실제 {model_type.upper()} 모델 훈련 시작!")
    
    # 설정 생성
    config = create_config(model_type)
    
    # 저장 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"experiments/real_{model_type}_{timestamp}"
    
    # 설정 저장
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 실제 모델 생성
    model = create_real_model(model_type, config)
    
    # 데이터 로더 생성 (기존 방식 사용)
    print("📂 데이터 로더 생성 중...")
    data_loader_wrapper = create_data_loaders(
        data_path=data_path,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        use_augmentation=config['data']['augmentation']['enabled'],
        stratify_by_sweetness=True,
        random_seed=42
    )
    
    # 개별 로더 추출
    train_loader = data_loader_wrapper.train_loader
    val_loader = data_loader_wrapper.val_loader
    test_loader = data_loader_wrapper.test_loader
    
    # 실제 모델 트레이너 생성 (기존 인프라 활용)
    trainer = RealModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir=save_dir,
        device=config['device']
    )
    
    # 훈련 실행
    results = trainer.train(
        num_epochs=config['training']['epochs'],
        save_every=5,
        validate_every=1,
        early_stopping=True,
        verbose=True
    )
    
    print(f"✅ {model_type.upper()} 모델 훈련 완료!")
    print(f"   🏆 최고 성능: Val MAE {results.get('best_val_mae', 'N/A')}")
    print(f"   📊 총 에포크: {results.get('total_epochs', 'N/A')}")
    
    return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Real Architecture Training (Simplified)")
    parser.add_argument("--model", choices=["efficientnet", "melspec_cnn", "both"], 
                       default="both", help="훈련할 모델")
    parser.add_argument("--data-path", default="watermelon_sound_data", 
                       help="데이터 경로")
    
    args = parser.parse_args()
    
    print("🍉 실제 아키텍처 훈련 시작!")
    print(f"   🎯 모델: {args.model}")
    print(f"   📂 데이터: {args.data_path}")
    print()
    
    results = {}
    
    try:
        if args.model in ["efficientnet", "both"]:
            results['efficientnet'] = train_real_model('efficientnet', args.data_path)
            print()
        
        if args.model in ["melspec_cnn", "both"]:
            results['melspec_cnn'] = train_real_model('melspec_cnn', args.data_path)
            print()
        
        # 최종 결과
        print("🎉 모든 훈련 완료!")
        for model_name, result in results.items():
            best_mae = result.get('best_val_mae', 'N/A')
            if isinstance(best_mae, (int, float)):
                print(f"   {model_name.upper()}: Best Val MAE = {best_mae:.4f}")
            else:
                print(f"   {model_name.upper()}: Best Val MAE = {best_mae}")
        
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 