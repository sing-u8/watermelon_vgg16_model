#!/usr/bin/env python3
"""
Real Model Training - Ultra Simple
매우 간단한 방식으로 실제 EfficientNet과 MelSpecCNN 훈련
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import create_trainer_from_config
from src.models.efficientnet_watermelon import create_efficientnet_watermelon
from src.models.melspec_cnn_watermelon import create_melspec_cnn_watermelon


def replace_model_in_trainer(trainer, new_model):
    """트레이너의 모델을 새로운 모델로 교체"""
    # 모델 교체
    trainer.model = new_model
    trainer.model.to(trainer.device)
    
    # 옵티마이저와 스케줄러 재생성 (간단한 방법)
    import torch.optim as optim
    
    # Adam 옵티마이저 재생성
    trainer.optimizer = optim.Adam(
        trainer.model.parameters(),
        lr=0.0001,
        weight_decay=1e-4
    )
    
    # StepLR 스케줄러 재생성
    trainer.scheduler = optim.lr_scheduler.StepLR(
        trainer.optimizer,
        step_size=10,
        gamma=0.7
    )
    
    print("✅ 트레이너 모델 교체 완료 (옵티마이저/스케줄러 재생성)")
    return trainer


def create_real_model_config(model_type: str) -> str:
    """실제 모델용 임시 설정 파일 생성"""
    config = {
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
        "model": {
            "input_channels": 3,
            "pretrained": False,  # VGG용이지만 무시됨
            "dropout_rate": 0.7,
            "freeze_features": False,
            "num_fc_layers": 2,
            "fc_hidden_size": 256
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
    
    # 임시 설정 파일 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = f"configs/temp_real_{model_type}_{timestamp}.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def train_real_model(model_type: str):
    """실제 모델 훈련"""
    print(f"🚀 실제 {model_type.upper()} 모델 훈련 시작!")
    
    # 임시 설정 파일 생성
    config_path = create_real_model_config(model_type)
    
    try:
        # 기존 트레이너 생성 (VGG 모델로 먼저 생성)
        print("📂 기본 트레이너 생성 중...")
        trainer = create_trainer_from_config(
            config_path=config_path,
            data_path="watermelon_sound_data",
            experiment_name=f"real_{model_type}_exp"
        )
        
        # 실제 모델 생성
        print(f"🔧 {model_type.upper()} 모델 생성 중...")
        if model_type == 'efficientnet':
            real_model = create_efficientnet_watermelon(
                pretrained=False,
                dropout_rate=0.7,
                num_fc_layers=2,
                fc_hidden_size=256
            )
        elif model_type == 'melspec_cnn':
            real_model = create_melspec_cnn_watermelon(
                input_channels=3,
                base_channels=32,
                dropout_rate=0.7
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
        
        # 모델 교체
        print("🔄 모델 교체 중...")
        trainer = replace_model_in_trainer(trainer, real_model)
        
        # 실제 모델 정보 출력
        real_model.print_model_info()
        
        # 훈련 실행
        print("🚀 훈련 시작!")
        results = trainer.train(
            num_epochs=30,
            save_every=5,
            validate_every=1,
            early_stopping=True,
            verbose=True
        )
        
        print(f"✅ {model_type.upper()} 모델 훈련 완료!")
        
        # 결과 출력
        best_val_mae = results.get('best_val_mae', 'N/A')
        if isinstance(best_val_mae, (int, float)):
            print(f"   🏆 최고 성능: Val MAE {best_val_mae:.4f}")
        else:
            print(f"   🏆 최고 성능: Val MAE {best_val_mae}")
        
        return results
        
    except Exception as e:
        print(f"❌ {model_type.upper()} 훈련 중 오류 발생: {e}")
        return None
        
    finally:
        # 임시 설정 파일 삭제
        if os.path.exists(config_path):
            os.remove(config_path)
            print(f"🗑️ 임시 설정 파일 삭제: {config_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Real Model Training (Ultra Simple)")
    parser.add_argument("--model", choices=["efficientnet", "melspec_cnn", "both"], 
                       default="both", help="훈련할 모델")
    
    args = parser.parse_args()
    
    print("🍉 실제 아키텍처 훈련 시작! (Ultra Simple)")
    print(f"   🎯 모델: {args.model}")
    print()
    
    results = {}
    success = True
    
    if args.model in ["efficientnet", "both"]:
        results['efficientnet'] = train_real_model('efficientnet')
        if results['efficientnet'] is None:
            success = False
        print()
    
    if args.model in ["melspec_cnn", "both"]:
        results['melspec_cnn'] = train_real_model('melspec_cnn')
        if results['melspec_cnn'] is None:
            success = False
        print()
    
    # 최종 결과
    if success:
        print("🎉 모든 훈련 완료!")
        for model_name, result in results.items():
            if result:
                best_mae = result.get('best_val_mae', 'N/A')
                if isinstance(best_mae, (int, float)):
                    print(f"   {model_name.upper()}: Best Val MAE = {best_mae:.4f}")
                else:
                    print(f"   {model_name.upper()}: Best Val MAE = {best_mae}")
    else:
        print("❌ 일부 훈련 실패")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 