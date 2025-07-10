#!/usr/bin/env python3
"""
🍉 Watermelon Ensemble Model Creation Script

기존 최고 성능 모델들을 조합하여 앙상블 모델을 생성하고 성능을 평가하는 스크립트
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 프로젝트 루트 디렉토리를 sys.path에 추가
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.vgg_watermelon import VGGWatermelon, create_vgg_watermelon
from training.data_loader import create_data_loaders
from training.trainer import create_trainer_from_config
from evaluation.evaluator import WatermelonEvaluator


class WatermelonEnsemble:
    """
    수박 당도 예측을 위한 앙상블 모델 클래스
    """
    
    def __init__(self, models: List[VGGWatermelon], weights: Optional[List[float]] = None):
        """
        앙상블 모델 초기화
        
        Args:
            models (List[VGGWatermelon]): 앙상블에 포함할 모델들
            weights (List[float], optional): 각 모델의 가중치
        """
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모든 모델을 같은 디바이스로 이동
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        print(f"🎯 앙상블 모델 생성 완료")
        print(f"   📊 모델 수: {len(self.models)}")
        print(f"   ⚖️ 가중치: {[f'{w:.3f}' for w in self.weights]}")
        print(f"   🖥️ 디바이스: {self.device}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        앙상블 예측 수행
        
        Args:
            x (torch.Tensor): 입력 텐서
            
        Returns:
            torch.Tensor: 앙상블 예측 결과
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        # 가중 평균 계산
        weighted_pred = sum(w * pred for w, pred in zip(self.weights, predictions))
        
        return weighted_pred
    
    def evaluate_ensemble(self, data_loader) -> Dict[str, Any]:
        """
        앙상블 모델 평가
        
        Args:
            data_loader: 데이터 로더
            
        Returns:
            Dict[str, Any]: 평가 메트릭들
        """
        all_predictions = []
        all_targets = []
        
        self.eval()
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 앙상블 예측
                predictions = self.predict(inputs)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 메트릭 계산
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # 허용 오차별 정확도 계산
        accuracy_05 = np.mean(np.abs(predictions - targets) <= 0.5) * 100
        accuracy_10 = np.mean(np.abs(predictions - targets) <= 1.0) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'accuracy_0.5': accuracy_05,
            'accuracy_1.0': accuracy_10,
            'predictions': predictions.tolist(),
            'targets': targets.tolist()
        }
    
    def eval(self):
        """모든 모델을 evaluation 모드로 설정"""
        for model in self.models:
            model.eval()


def train_individual_models(experiment_configs: List[str], 
                          data_path: str,
                          quick_training: bool = True) -> List[str]:
    """
    개별 모델들을 훈련시키고 체크포인트 경로 반환
    
    Args:
        experiment_configs (List[str]): 실험 설정 파일 경로들
        data_path (str): 데이터 경로
        quick_training (bool): 빠른 훈련 모드 (에포크 수 줄임)
        
    Returns:
        List[str]: 훈련된 모델 체크포인트 경로들
    """
    model_paths = []
    
    for i, config_path in enumerate(experiment_configs):
        print(f"\n🏋️ 모델 {i+1}/{len(experiment_configs)} 훈련 시작")
        print(f"   📄 설정: {config_path}")
        
        # 실험 이름 생성
        config_name = Path(config_path).stem
        experiment_name = f"ensemble_{config_name}_{int(time.time())}"
        
        # 빠른 훈련을 위해 설정 수정
        if quick_training:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # 에포크 수 줄이기
            config['training']['epochs'] = min(10, config['training'].get('epochs', 100))
            config['early_stopping']['patience'] = 5
            
            # 임시 설정 파일 생성
            temp_config_path = f"configs/temp_{config_name}.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f)
            
            config_path = temp_config_path
        
        try:
            # 트레이너 생성 및 훈련
            trainer = create_trainer_from_config(config_path, data_path, experiment_name)
            results = trainer.train(
                num_epochs=10 if quick_training else None,
                verbose=False
            )
            
            # 최고 성능 모델 경로 저장
            model_path = trainer.best_model_path
            if Path(model_path).exists():
                model_paths.append(str(model_path))
                print(f"   ✅ 훈련 완료 - MAE: {results['best_val_mae']:.3f}")
            else:
                print(f"   ❌ 모델 파일을 찾을 수 없음: {model_path}")
        
        except Exception as e:
            print(f"   ❌ 훈련 실패: {str(e)}")
            continue
        
        finally:
            # 임시 설정 파일 삭제
            if quick_training and 'temp_' in config_path:
                Path(config_path).unlink(missing_ok=True)
    
    return model_paths


def create_ensemble_from_existing_configs(data_path: str, save_dir: str = "experiments/ensemble") -> Dict[str, Any]:
    """
    기존 성공한 실험 설정들로부터 앙상블 모델 생성
    
    Args:
        data_path (str): 데이터 경로
        save_dir (str): 결과 저장 디렉토리
        
    Returns:
        Dict[str, Any]: 앙상블 결과
    """
    print("🎯 앙상블 모델 생성 시작")
    
    # 기존 성공한 실험 설정들
    successful_configs = [
        "configs/training_batch16.yaml",    # Val MAE: 0.7103, Test MAE: 1.5267 (최고 일반화)
        "configs/training_batch32.yaml",    # Val MAE: 0.5462, Test MAE: 1.7936 (최고 검증)
        "configs/training_huber_loss.yaml", # Val MAE: 0.6119, Test MAE: 1.7399 (강건성)
    ]
    
    # 설정 파일 존재 확인
    existing_configs = []
    for config_path in successful_configs:
        if Path(config_path).exists():
            existing_configs.append(config_path)
        else:
            print(f"⚠️ 설정 파일 없음: {config_path}")
    
    if not existing_configs:
        raise FileNotFoundError("사용 가능한 설정 파일이 없습니다.")
    
    print(f"✅ 사용할 설정 파일: {len(existing_configs)}개")
    
    # 개별 모델들 훈련
    print("\n🏋️ 개별 모델 훈련 시작...")
    model_paths = train_individual_models(existing_configs, data_path, quick_training=True)
    
    if not model_paths:
        raise RuntimeError("훈련된 모델이 없습니다.")
    
    print(f"✅ 훈련된 모델: {len(model_paths)}개")
    
    # 데이터 로더 생성
    data_loader = create_data_loaders(data_path, config_path=existing_configs[0])
    
    # 개별 모델 성능 평가
    print("\n📊 개별 모델 성능 평가...")
    individual_results = {}
    models = []
    
    for i, model_path in enumerate(model_paths):
        try:
            # 모델 로드 (PyTorch 2.6 호환성을 위해 weights_only=False 명시)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model_config = checkpoint.get('model_config', {})
            
            # VGGWatermelon이 지원하는 파라미터만 필터링
            supported_params = [
                'input_channels', 'pretrained', 'dropout_rate', 
                'freeze_features', 'num_fc_layers', 'fc_hidden_size'
            ]
            filtered_config = {k: v for k, v in model_config.items() if k in supported_params}
            
            model = create_vgg_watermelon(**filtered_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            models.append(model)
            
            # 개별 모델 평가
            ensemble = WatermelonEnsemble([model])
            results = ensemble.evaluate_ensemble(data_loader.test_loader)
            
            model_name = f"Model_{i+1}"
            individual_results[model_name] = results
            
            print(f"   {model_name}: MAE={results['mae']:.3f}, R²={results['r2_score']:.3f}")
            
        except Exception as e:
            print(f"   ❌ 모델 {i+1} 평가 실패: {str(e)}")
            continue
    
    if not models:
        raise RuntimeError("로드된 모델이 없습니다.")
    
    # 앙상블 모델 생성 및 평가
    print("\n🎯 앙상블 모델 평가...")
    
    # 1. Equal Weight Ensemble
    equal_ensemble = WatermelonEnsemble(models)
    equal_results = equal_ensemble.evaluate_ensemble(data_loader.test_loader)
    
    # 2. Performance-based Weighted Ensemble
    weights = []
    for model_name, results in individual_results.items():
        # MAE가 낮을수록 높은 가중치 (1/MAE 정규화)
        weight = 1.0 / results['mae']
        weights.append(weight)
    
    # 가중치 정규화
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    weighted_ensemble = WatermelonEnsemble(models, weights)
    weighted_results = weighted_ensemble.evaluate_ensemble(data_loader.test_loader)
    
    # 결과 정리
    ensemble_results = {
        'individual_models': individual_results,
        'equal_weight_ensemble': equal_results,
        'weighted_ensemble': weighted_results,
        'weights': weights,
        'model_count': len(models)
    }
    
    # 결과 출력
    print("\n📊 앙상블 결과:")
    print(f"   🟰 Equal Weight - MAE: {equal_results['mae']:.3f}, "
          f"Acc(±0.5): {equal_results['accuracy_0.5']:.1f}%, "
          f"Acc(±1.0): {equal_results['accuracy_1.0']:.1f}%")
    print(f"   ⚖️ Weighted     - MAE: {weighted_results['mae']:.3f}, "
          f"Acc(±0.5): {weighted_results['accuracy_0.5']:.1f}%, "
          f"Acc(±1.0): {weighted_results['accuracy_1.0']:.1f}%")
    
    # 결과 저장
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    # JSON 결과 저장
    results_path = save_dir_path / "ensemble_results.json"
    with open(results_path, 'w') as f:
        json.dump(ensemble_results, f, indent=2)
    
    # 시각화
    create_ensemble_visualization(ensemble_results, save_dir_path)
    
    print(f"\n💾 결과 저장됨: {save_dir}")
    
    return ensemble_results


def create_ensemble_visualization(results: Dict[str, Any], save_dir: Path):
    """
    앙상블 결과 시각화
    
    Args:
        results (Dict): 앙상블 결과
        save_dir (Path): 저장 디렉토리
    """
    plt.style.use('default')
    
    # 1. 모델 성능 비교
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MAE 비교
    models = list(results['individual_models'].keys()) + ['Equal Ensemble', 'Weighted Ensemble']
    mae_values = [results['individual_models'][m]['mae'] for m in results['individual_models'].keys()]
    mae_values += [results['equal_weight_ensemble']['mae'], results['weighted_ensemble']['mae']]
    
    colors = ['lightblue'] * len(results['individual_models']) + ['orange', 'red']
    
    axes[0, 0].bar(range(len(models)), mae_values, color=colors)
    axes[0, 0].set_title('Model Performance Comparison (MAE)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Mean Absolute Error (Brix)')
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy 비교 (±0.5 Brix)
    acc_05_values = [results['individual_models'][m]['accuracy_0.5'] for m in results['individual_models'].keys()]
    acc_05_values += [results['equal_weight_ensemble']['accuracy_0.5'], results['weighted_ensemble']['accuracy_0.5']]
    
    axes[0, 1].bar(range(len(models)), acc_05_values, color=colors)
    axes[0, 1].set_title('Accuracy Comparison (±0.5 Brix)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 예측 vs 실제 (Weighted Ensemble)
    predictions = np.array(results['weighted_ensemble']['predictions'])
    targets = np.array(results['weighted_ensemble']['targets'])
    
    axes[1, 0].scatter(targets, predictions, alpha=0.6, color='red')
    axes[1, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', lw=2)
    axes[1, 0].set_xlabel('Actual Sweetness (Brix)')
    axes[1, 0].set_ylabel('Predicted Sweetness (Brix)')
    axes[1, 0].set_title('Weighted Ensemble: Prediction vs Actual', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 오차 분포
    errors = predictions - targets
    # 1차원 배열로 변환하여 단일 데이터셋으로 인식하게 함
    errors_flat = errors.flatten()
    axes[1, 1].hist(errors_flat, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Prediction Error (Brix)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Weighted Ensemble: Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "ensemble_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 시각화 저장됨: ensemble_performance.png")


def optimize_preprocessing_parameters(data_path: str, save_dir: str = "experiments/preprocessing_optimization"):
    """
    전처리 파라미터 최적화 실험
    
    Args:
        data_path (str): 데이터 경로
        save_dir (str): 결과 저장 디렉토리
    """
    print("\n🔧 전처리 파라미터 최적화 시작")
    
    # 다양한 전처리 설정들
    preprocessing_configs = [
        {
            'name': 'baseline',
            'n_mels': 128,
            'n_fft': 2048,
            'hop_length': 512,
            'description': '기본 설정'
        },
        {
            'name': 'high_resolution',
            'n_mels': 256,
            'n_fft': 4096,
            'hop_length': 512,
            'description': '고해상도 스펙트로그램'
        },
        {
            'name': 'low_complexity',
            'n_mels': 64,
            'n_fft': 1024,
            'hop_length': 256,
            'description': '저복잡도 설정'
        },
        {
            'name': 'fine_temporal',
            'n_mels': 128,
            'n_fft': 2048,
            'hop_length': 256,
            'description': '세밀한 시간 해상도'
        }
    ]
    
    results = {}
    
    for config in preprocessing_configs:
        print(f"\n📊 {config['name']} 설정 테스트 중...")
        print(f"   📝 설명: {config['description']}")
        print(f"   🔧 파라미터: n_mels={config['n_mels']}, n_fft={config['n_fft']}, hop_length={config['hop_length']}")
        
        try:
            # 임시 데이터 설정 파일 생성
            temp_data_config = {
                'spectrogram': {
                    'n_mels': config['n_mels'],
                    'n_fft': config['n_fft'],
                    'hop_length': config['hop_length'],
                    'resize': {'enabled': True, 'height': 224, 'width': 224}
                }
            }
            
            temp_config_path = f"configs/temp_data_{config['name']}.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(temp_data_config, f)
            
            # 빠른 모델 훈련 및 평가
            trainer_config = "configs/training_batch16.yaml"  # 가장 빠른 설정 사용
            
            with open(trainer_config, 'r') as f:
                training_config = yaml.safe_load(f)
            
            # 빠른 훈련을 위한 설정 수정
            training_config['training']['epochs'] = 5
            training_config['early_stopping']['patience'] = 3
            
            temp_training_config = f"configs/temp_training_{config['name']}.yaml"
            with open(temp_training_config, 'w') as f:
                yaml.dump(training_config, f)
            
            # 모델 훈련
            experiment_name = f"preprocessing_{config['name']}_{int(time.time())}"
            trainer = create_trainer_from_config(temp_training_config, data_path, experiment_name)
            
            training_results = trainer.train(num_epochs=5, verbose=False)
            
            results[config['name']] = {
                'config': config,
                'val_mae': training_results['best_val_mae'],
                'test_mae': training_results['final_test_metrics'].get('mae', 'N/A'),
                'training_time': training_results['training_time']
            }
            
            print(f"   ✅ 완료 - Val MAE: {training_results['best_val_mae']:.3f}")
            
            # 임시 파일 정리
            Path(temp_config_path).unlink(missing_ok=True)
            Path(temp_training_config).unlink(missing_ok=True)
            
        except Exception as e:
            print(f"   ❌ 실패: {str(e)}")
            continue
    
    # 결과 정리 및 저장
    if results:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        
        # 결과 저장
        results_path = save_dir_path / "preprocessing_optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 최고 성능 설정 찾기
        best_config = min(results.items(), key=lambda x: x[1]['val_mae'])
        
        print(f"\n🏆 최고 성능 전처리 설정: {best_config[0]}")
        print(f"   📊 Val MAE: {best_config[1]['val_mae']:.3f}")
        print(f"   🔧 파라미터: {best_config[1]['config']}")
        
        print(f"\n💾 전처리 최적화 결과 저장됨: {save_dir}")
        
        return results
    else:
        print("❌ 전처리 최적화 실험에서 유효한 결과가 없습니다.")
        return {}


def main():
    """메인 실행 함수"""
    print("🍉 수박 당도 예측 - 앙상블 모델링 & 전처리 최적화")
    print("=" * 60)
    
    # 설정
    data_path = "watermelon_sound_data"
    base_save_dir = "experiments"
    
    # 명령행 인자 확인 (전처리 최적화만 실행할지 여부)
    import sys
    preprocessing_only = "--preprocessing-only" in sys.argv
    
    try:
        ensemble_results = {}
        
        if not preprocessing_only:
            # 1. 앙상블 모델링
            print("\n🎯 1단계: 앙상블 모델링")
            ensemble_results = create_ensemble_from_existing_configs(
                data_path, 
                f"{base_save_dir}/ensemble_optimization"
            )
        else:
            print("\n⚡ 전처리 최적화만 실행합니다...")
        
        # 2. 전처리 파라미터 최적화
        print("\n🔧 2단계: 전처리 파라미터 최적화")
        preprocessing_results = optimize_preprocessing_parameters(
            data_path, 
            f"{base_save_dir}/preprocessing_optimization"
        )
        
        # 3. 종합 결과 정리
        print("\n📋 3단계: 종합 결과 정리")
        
        if ensemble_results and 'individual_models' in ensemble_results:
            # 최고 성능 추출
            best_individual = min(ensemble_results['individual_models'].items(), 
                                key=lambda x: x[1]['mae'])
            
            equal_ensemble = ensemble_results['equal_weight_ensemble']
            weighted_ensemble = ensemble_results['weighted_ensemble']
            
            print("\n🏆 최종 성능 비교:")
            print(f"   🥉 최고 개별 모델: MAE={best_individual[1]['mae']:.3f}, Acc(±0.5)={best_individual[1]['accuracy_0.5']:.1f}%")
            print(f"   🥈 Equal Ensemble: MAE={equal_ensemble['mae']:.3f}, Acc(±0.5)={equal_ensemble['accuracy_0.5']:.1f}%")
            print(f"   🥇 Weighted Ensemble: MAE={weighted_ensemble['mae']:.3f}, Acc(±0.5)={weighted_ensemble['accuracy_0.5']:.1f}%")
            
            # 90% 정확도 목표 달성 여부 확인
            target_accuracy_90 = 90.0
            
            print(f"\n🎯 90% 정확도 목표 달성 분석:")
            for name, acc in [
                ("최고 개별 모델", best_individual[1]['accuracy_0.5']),
                ("Equal Ensemble", equal_ensemble['accuracy_0.5']),
                ("Weighted Ensemble", weighted_ensemble['accuracy_0.5'])
            ]:
                status = "✅ 달성" if acc >= target_accuracy_90 else f"❌ 미달 ({target_accuracy_90 - acc:.1f}%p 부족)"
                print(f"   {name}: {acc:.1f}% - {status}")
        else:
            print("📝 앙상블 결과가 없습니다 (전처리 최적화만 실행됨)")
        
        # 전처리 최적화 결과
        if preprocessing_results:
            best_preprocessing = min(preprocessing_results.items(), key=lambda x: x[1]['val_mae'])
            print(f"\n🔧 최적 전처리 설정: {best_preprocessing[0]} (MAE: {best_preprocessing[1]['val_mae']:.3f})")
        
        print("\n✅ 모든 최적화 작업 완료!")
        
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 