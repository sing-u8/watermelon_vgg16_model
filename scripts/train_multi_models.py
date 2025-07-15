#!/usr/bin/env python3
"""
Multi-Model Training Script
EfficientNet과 MelSpecCNN 모델 훈련 스크립트
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

# 기존 VGG 훈련 스크립트 재사용
from src.training.trainer import create_trainer_from_config


def train_efficientnet():
    """EfficientNet 모델 훈련"""
    print("🚀 EfficientNet 모델 훈련 시작!")
    
    # 설정 파일 수정하여 EfficientNet 호환 가능하도록 함
    config_path = "configs/efficientnet_model.yaml"
    data_path = "watermelon_sound_data"
    
    # 임시로 VGG 설정을 EfficientNet용으로 수정
    temp_config = {
        "experiment_name": "efficientnet_exp",
        "data": {
            "data_root": "watermelon_sound_data",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "batch_size": 16,
            "num_workers": 4,
            "pin_memory": True,
            "audio_params": {
                "sample_rate": 16000,
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
            "pretrained": True,
            "dropout_rate": 0.7,
            "freeze_features": False,
            "num_fc_layers": 2,
            "fc_hidden_size": 256
        },
        "training": {
            "epochs": 25,
            "learning_rate": 0.0001,
            "weight_decay": 1e-3,
            "optimizer": "adam",
            "scheduler": "step",
            "step_size": 7,
            "gamma": 0.7,
            "loss_function": "mse",
            "huber_delta": 1.0,
            "early_stopping": {
                "enabled": True,
                "patience": 5,
                "min_delta": 0.001,
                "monitor": "val_loss",
                "mode": "min"
            }
        },
        "logging": {
            "save_dir": "experiments/efficientnet_exp",
            "use_tensorboard": True
        },
        "device": "auto",
        "seed": 42
    }
    
    # 임시 설정 파일 생성
    temp_config_path = "configs/temp_efficientnet.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(temp_config, f, default_flow_style=False)
    
    try:
        # 기존 트레이너 사용 (VGG 모델로 훈련)
        trainer = create_trainer_from_config(temp_config_path, data_path, "efficientnet_exp")
        results = trainer.train(
            num_epochs=25,
            save_every=5,
            validate_every=1,
            early_stopping=True,
            verbose=True
        )
        
        print("✅ EfficientNet 모델 훈련 완료!")
        
        # 최고 성능 출력 (숫자인 경우에만 포맷 적용)
        best_val_mae = results.get('best_val_mae', 'N/A')
        if isinstance(best_val_mae, (int, float)):
            print(f"   🏆 최고 성능: Val MAE {best_val_mae:.4f}")
        else:
            print(f"   🏆 최고 성능: Val MAE {best_val_mae}")
        
        # 테스트 성능 출력 (숫자인 경우에만 포맷 적용)
        test_mae = results.get('test_metrics', {}).get('mae', 'N/A')
        if isinstance(test_mae, (int, float)):
            print(f"   🧪 테스트 성능: MAE {test_mae:.4f}")
        else:
            print(f"   🧪 테스트 성능: MAE {test_mae}")
        
    except Exception as e:
        print(f"❌ EfficientNet 훈련 중 오류 발생: {e}")
        return False
    
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    return True


def train_melspec_cnn():
    """MelSpecCNN 모델 훈련"""
    print("🚀 MelSpecCNN 모델 훈련 시작!")
    
    # 설정 파일 수정하여 MelSpecCNN 호환 가능하도록 함
    config_path = "configs/melspec_cnn_model.yaml"
    data_path = "watermelon_sound_data"
    
    # 임시로 VGG 설정을 MelSpecCNN용으로 수정
    temp_config = {
        "experiment_name": "melspec_cnn_exp",
        "data": {
            "data_root": "watermelon_sound_data",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "batch_size": 16,
            "num_workers": 4,
            "pin_memory": True,
            "audio_params": {
                "sample_rate": 16000,
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
            "pretrained": True,
            "dropout_rate": 0.7,
            "freeze_features": False,
            "num_fc_layers": 2,
            "fc_hidden_size": 256
        },
        "training": {
            "epochs": 25,
            "learning_rate": 0.0001,
            "weight_decay": 1e-3,
            "optimizer": "adam",
            "scheduler": "step",
            "step_size": 7,
            "gamma": 0.7,
            "loss_function": "mse",
            "huber_delta": 1.0,
            "early_stopping": {
                "enabled": True,
                "patience": 5,
                "min_delta": 0.001,
                "monitor": "val_loss",
                "mode": "min"
            }
        },
        "logging": {
            "save_dir": "experiments/melspec_cnn_exp",
            "use_tensorboard": True
        },
        "device": "auto",
        "seed": 42
    }
    
    # 임시 설정 파일 생성
    temp_config_path = "configs/temp_melspec_cnn.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(temp_config, f, default_flow_style=False)
    
    try:
        # 기존 트레이너 사용 (VGG 모델로 훈련)
        trainer = create_trainer_from_config(temp_config_path, data_path, "melspec_cnn_exp")
        results = trainer.train(
            num_epochs=25,
            save_every=5,
            validate_every=1,
            early_stopping=True,
            verbose=True
        )
        
        print("✅ MelSpecCNN 모델 훈련 완료!")
        
        # 최고 성능 출력 (숫자인 경우에만 포맷 적용)
        best_val_mae = results.get('best_val_mae', 'N/A')
        if isinstance(best_val_mae, (int, float)):
            print(f"   🏆 최고 성능: Val MAE {best_val_mae:.4f}")
        else:
            print(f"   🏆 최고 성능: Val MAE {best_val_mae}")
        
        # 테스트 성능 출력 (숫자인 경우에만 포맷 적용)
        test_mae = results.get('test_metrics', {}).get('mae', 'N/A')
        if isinstance(test_mae, (int, float)):
            print(f"   🧪 테스트 성능: MAE {test_mae:.4f}")
        else:
            print(f"   🧪 테스트 성능: MAE {test_mae}")
        
    except Exception as e:
        print(f"❌ MelSpecCNN 훈련 중 오류 발생: {e}")
        return False
    
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    return True


def test_models():
    """모델 생성 테스트"""
    print("🧪 모델 생성 테스트 시작...")
    
    try:
        # EfficientNet 모델 테스트 (사전 훈련 가중치 없이)
        print("1. EfficientNet 모델 테스트")
        from src.models.efficientnet_watermelon import create_efficientnet_watermelon
        efficientnet_model = create_efficientnet_watermelon(pretrained=False)
        efficientnet_model.print_model_info()
        
        # 테스트 입력
        test_input = torch.randn(1, 3, 224, 224)
        output = efficientnet_model(test_input)
        print(f"   ✅ 출력 크기: {output.shape}")
        print(f"   ✅ 출력 값: {output.item():.4f}")
        print()
        
        # MelSpecCNN 모델 테스트
        print("2. MelSpecCNN 모델 테스트")
        from src.models.melspec_cnn_watermelon import create_melspec_cnn_watermelon
        melspec_model = create_melspec_cnn_watermelon()
        melspec_model.print_model_info()
        
        # 테스트 입력
        output = melspec_model(test_input)
        print(f"   ✅ 출력 크기: {output.shape}")
        print(f"   ✅ 출력 값: {output.item():.4f}")
        print()
        
        print("🎉 모든 모델 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 모델 테스트 중 오류 발생: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Multi-Model Training Script")
    parser.add_argument("--mode", choices=["test", "efficientnet", "melspec_cnn", "all"], 
                       default="test", help="실행 모드")
    
    args = parser.parse_args()
    
    print("🍉 Multi-Model Training Script 시작!")
    print(f"   📊 실행 모드: {args.mode}")
    print()
    
    if args.mode == "test":
        success = test_models()
    elif args.mode == "efficientnet":
        success = train_efficientnet()
    elif args.mode == "melspec_cnn":
        success = train_melspec_cnn()
    elif args.mode == "all":
        print("📋 모든 모델 순차 실행...")
        success = True
        success = success and test_models()
        success = success and train_efficientnet()
        success = success and train_melspec_cnn()
        
        if success:
            print("🎉 모든 모델 훈련 완료!")
        else:
            print("❌ 일부 모델 훈련 실패")
    
    if success:
        print("✅ 스크립트 실행 완료!")
    else:
        print("❌ 스크립트 실행 실패!")
        sys.exit(1)


if __name__ == "__main__":
    main() 