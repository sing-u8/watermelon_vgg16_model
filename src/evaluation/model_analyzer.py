"""
Model Analyzer
모델의 상세 분석 및 비교를 위한 모듈
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import json
import time
from tqdm import tqdm

# 상대 import
import sys
from pathlib import Path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from models.vgg_watermelon import VGGWatermelon
from training.data_loader import WatermelonDataLoader


class ModelAnalyzer:
    """
    모델의 상세 분석을 위한 클래스
    """
    
    def __init__(self, model: VGGWatermelon, device: torch.device):
        """
        ModelAnalyzer 초기화
        
        Args:
            model (VGGWatermelon): 분석할 모델
            device (torch.device): 디바이스
        """
        self.model = model.to(device)
        self.device = device
        
        print(f"🔬 ModelAnalyzer 초기화 완료")
        print(f"   🖥️ 디바이스: {self.device}")
    
    def analyze_model_architecture(self) -> Dict[str, Any]:
        """
        모델 아키텍처 분석
        
        Returns:
            Dict[str, Any]: 아키텍처 분석 결과
        """
        print("🏗️ 모델 아키텍처 분석 중...")
        
        # 기본 모델 정보
        model_info = self.model.get_model_info()
        
        # 레이어별 파라미터 분석
        layer_analysis = self._analyze_layers()
        
        # 메모리 사용량 분석
        memory_analysis = self._analyze_memory_usage()
        
        # 계산 복잡도 분석
        computational_analysis = self._analyze_computational_complexity()
        
        return {
            'model_info': model_info,
            'layer_analysis': layer_analysis,
            'memory_analysis': memory_analysis,
            'computational_analysis': computational_analysis
        }
    
    def _analyze_layers(self) -> Dict[str, Any]:
        """레이어별 파라미터 분석"""
        layer_info = {}
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # leaf module
                params = sum(p.numel() for p in module.parameters())
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                layer_info[name] = {
                    'type': module.__class__.__name__,
                    'parameters': params,
                    'trainable_parameters': trainable_params,
                    'trainable_ratio': trainable_params / params if params > 0 else 0
                }
        
        return layer_info
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 분석"""
        # 더미 입력으로 메모리 사용량 측정
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 순전파
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
            max_memory = torch.cuda.max_memory_allocated() / 1024**2   # MB
            
            return {
                'memory_allocated_mb': memory_allocated,
                'memory_reserved_mb': memory_reserved,
                'max_memory_allocated_mb': max_memory,
                'device': str(self.device)
            }
        else:
            return {
                'memory_allocated_mb': 0,
                'memory_reserved_mb': 0,
                'max_memory_allocated_mb': 0,
                'device': str(self.device),
                'note': 'CPU memory analysis not available'
            }
    
    def _analyze_computational_complexity(self) -> Dict[str, Any]:
        """계산 복잡도 분석"""
        # FLOPs 계산을 위한 간단한 추정
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # 더미 입력으로 추론 시간 측정
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # 워밍업
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # 실제 측정
        inference_times = []
        with torch.no_grad():
            for _ in range(100):
                start_time = time.time()
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                inference_times.append(time.time() - start_time)
        
        return {
            'total_parameters': total_params,
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'std_inference_time_ms': np.std(inference_times) * 1000,
            'min_inference_time_ms': np.min(inference_times) * 1000,
            'max_inference_time_ms': np.max(inference_times) * 1000,
            'throughput_fps': 1.0 / np.mean(inference_times)
        }
    
    def analyze_feature_maps(self, 
                           data_loader: WatermelonDataLoader,
                           num_samples: int = 10) -> Dict[str, Any]:
        """
        특성 맵 분석
        
        Args:
            data_loader (WatermelonDataLoader): 데이터 로더
            num_samples (int): 분석할 샘플 수
            
        Returns:
            Dict[str, Any]: 특성 맵 분석 결과
        """
        print(f"🗺️ 특성 맵 분석 중 ({num_samples}개 샘플)...")
        
        self.model.eval()
        feature_stats = {}
        
        sample_count = 0
        with torch.no_grad():
            for inputs, targets in data_loader.test_loader:
                if sample_count >= num_samples:
                    break
                
                inputs = inputs.to(self.device)
                
                # 여러 레이어에서 특성 맵 추출
                layer_indices = [5, 10, 15, 20, 25]  # VGG-16의 주요 레이어들
                
                for layer_idx in layer_indices:
                    try:
                        feature_map = self.model.get_feature_maps(inputs, layer_idx)
                        
                        if f'layer_{layer_idx}' not in feature_stats:
                            feature_stats[f'layer_{layer_idx}'] = {
                                'activations': [],
                                'shape': feature_map.shape[1:],  # 배치 차원 제외
                                'layer_index': layer_idx
                            }
                        
                        # 평균 활성화 값 저장
                        avg_activation = feature_map.mean(dim=(2, 3)).cpu().numpy()  # 공간 차원 평균
                        feature_stats[f'layer_{layer_idx}']['activations'].append(avg_activation)
                        
                    except Exception as e:
                        print(f"Warning: 레이어 {layer_idx} 특성 맵 추출 실패: {e}")
                
                sample_count += inputs.size(0)
        
        # 통계 계산
        for layer_name, stats in feature_stats.items():
            activations = np.concatenate(stats['activations'], axis=0)
            
            stats['statistics'] = {
                'mean_activation': float(np.mean(activations)),
                'std_activation': float(np.std(activations)),
                'min_activation': float(np.min(activations)),
                'max_activation': float(np.max(activations)),
                'num_channels': activations.shape[1],
                'sparsity': float(np.mean(activations == 0))  # 0인 활성화 비율
            }
            
            # 메모리 절약을 위해 원본 데이터 삭제
            del stats['activations']
        
        return feature_stats
    
    def analyze_gradients(self, 
                         data_loader: WatermelonDataLoader,
                         num_samples: int = 10) -> Dict[str, Any]:
        """
        그래디언트 분석
        
        Args:
            data_loader (WatermelonDataLoader): 데이터 로더
            num_samples (int): 분석할 샘플 수
            
        Returns:
            Dict[str, Any]: 그래디언트 분석 결과
        """
        print(f"📈 그래디언트 분석 중 ({num_samples}개 샘플)...")
        
        self.model.train()  # 그래디언트 계산을 위해 train 모드
        
        gradient_stats = {}
        loss_fn = nn.MSELoss()
        
        sample_count = 0
        for inputs, targets in data_loader.val_loader:
            if sample_count >= num_samples:
                break
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 그래디언트 초기화
            self.model.zero_grad()
            
            # 순전파 및 역전파
            predictions = self.model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            
            # 그래디언트 통계 수집
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach().cpu().numpy()
                    
                    if name not in gradient_stats:
                        gradient_stats[name] = {
                            'gradients': [],
                            'parameter_shape': param.shape
                        }
                    
                    gradient_stats[name]['gradients'].append(grad.flatten())
            
            sample_count += inputs.size(0)
        
        # 그래디언트 통계 계산
        for param_name, stats in gradient_stats.items():
            gradients = np.concatenate(stats['gradients'])
            
            stats['statistics'] = {
                'mean_gradient': float(np.mean(gradients)),
                'std_gradient': float(np.std(gradients)),
                'min_gradient': float(np.min(gradients)),
                'max_gradient': float(np.max(gradients)),
                'gradient_norm': float(np.linalg.norm(gradients)),
                'zero_gradient_ratio': float(np.mean(gradients == 0))
            }
            
            # 메모리 절약
            del stats['gradients']
        
        self.model.eval()  # 분석 후 eval 모드로 복귀
        
        return gradient_stats
    
    def analyze_prediction_confidence(self, 
                                    data_loader: WatermelonDataLoader,
                                    num_bootstrap: int = 100) -> Dict[str, Any]:
        """
        예측 신뢰도 분석 (Bootstrap 기반)
        
        Args:
            data_loader (WatermelonDataLoader): 데이터 로더
            num_bootstrap (int): Bootstrap 반복 횟수
            
        Returns:
            Dict[str, Any]: 신뢰도 분석 결과
        """
        print(f"🎯 예측 신뢰도 분석 중 (Bootstrap: {num_bootstrap}회)...")
        
        self.model.eval()
        
        # 전체 테스트 데이터에 대한 예측 수집
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader.test_loader:
                inputs = inputs.to(self.device)
                predictions = self.model(inputs)
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(targets.numpy().flatten())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Bootstrap 분석
        bootstrap_results = []
        n_samples = len(all_predictions)
        
        for _ in tqdm(range(num_bootstrap), desc="Bootstrap 분석"):
            # 복원 추출
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_preds = all_predictions[indices]
            boot_targets = all_targets[indices]
            
            # 메트릭 계산
            mae = np.mean(np.abs(boot_preds - boot_targets))
            rmse = np.sqrt(np.mean((boot_preds - boot_targets) ** 2))
            r2 = 1 - np.sum((boot_targets - boot_preds) ** 2) / np.sum((boot_targets - np.mean(boot_targets)) ** 2)
            
            bootstrap_results.append({
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            })
        
        # Bootstrap 통계
        bootstrap_df = pd.DataFrame(bootstrap_results)
        
        confidence_analysis = {
            'bootstrap_statistics': {
                'mae': {
                    'mean': float(bootstrap_df['mae'].mean()),
                    'std': float(bootstrap_df['mae'].std()),
                    'ci_lower': float(bootstrap_df['mae'].quantile(0.025)),
                    'ci_upper': float(bootstrap_df['mae'].quantile(0.975))
                },
                'rmse': {
                    'mean': float(bootstrap_df['rmse'].mean()),
                    'std': float(bootstrap_df['rmse'].std()),
                    'ci_lower': float(bootstrap_df['rmse'].quantile(0.025)),
                    'ci_upper': float(bootstrap_df['rmse'].quantile(0.975))
                },
                'r2': {
                    'mean': float(bootstrap_df['r2'].mean()),
                    'std': float(bootstrap_df['r2'].std()),
                    'ci_lower': float(bootstrap_df['r2'].quantile(0.025)),
                    'ci_upper': float(bootstrap_df['r2'].quantile(0.975))
                }
            },
            'prediction_uncertainty': {
                'mean_prediction_std': float(np.std(all_predictions)),
                'prediction_range': float(np.max(all_predictions) - np.min(all_predictions)),
                'coefficient_of_variation': float(np.std(all_predictions) / np.mean(all_predictions))
            },
            'num_bootstrap_samples': num_bootstrap,
            'total_test_samples': n_samples
        }
        
        return confidence_analysis
    
    def benchmark_inference_speed(self, 
                                 batch_sizes: List[int] = [1, 4, 8, 16, 32],
                                 num_runs: int = 100) -> Dict[str, Any]:
        """
        다양한 배치 크기에서 추론 속도 벤치마크
        
        Args:
            batch_sizes (List[int]): 테스트할 배치 크기들
            num_runs (int): 각 배치 크기별 실행 횟수
            
        Returns:
            Dict[str, Any]: 벤치마크 결과
        """
        print(f"⚡ 추론 속도 벤치마크 중...")
        
        self.model.eval()
        benchmark_results = {}
        
        for batch_size in batch_sizes:
            print(f"   배치 크기 {batch_size} 테스트 중...")
            
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
            
            # 워밍업
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_input)
            
            # 실제 벤치마크
            inference_times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = self.model(dummy_input)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    inference_times.append(time.time() - start_time)
            
            benchmark_results[f'batch_size_{batch_size}'] = {
                'batch_size': batch_size,
                'avg_time_ms': np.mean(inference_times) * 1000,
                'std_time_ms': np.std(inference_times) * 1000,
                'min_time_ms': np.min(inference_times) * 1000,
                'max_time_ms': np.max(inference_times) * 1000,
                'throughput_samples_per_sec': batch_size / np.mean(inference_times),
                'time_per_sample_ms': np.mean(inference_times) * 1000 / batch_size
            }
        
        return benchmark_results
    
    def generate_analysis_report(self, 
                               data_loader: WatermelonDataLoader,
                               save_path: str = "model_analysis_report.json") -> str:
        """
        종합 분석 보고서 생성
        
        Args:
            data_loader (WatermelonDataLoader): 데이터 로더
            save_path (str): 보고서 저장 경로
            
        Returns:
            str: 저장된 보고서 경로
        """
        print("📊 종합 분석 보고서 생성 중...")
        
        # 각종 분석 수행
        architecture_analysis = self.analyze_model_architecture()
        feature_analysis = self.analyze_feature_maps(data_loader, num_samples=20)
        gradient_analysis = self.analyze_gradients(data_loader, num_samples=10)
        confidence_analysis = self.analyze_prediction_confidence(data_loader, num_bootstrap=50)
        benchmark_results = self.benchmark_inference_speed()
        
        # 종합 보고서
        comprehensive_report = {
            'model_analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'architecture_analysis': architecture_analysis,
            'feature_map_analysis': feature_analysis,
            'gradient_analysis': gradient_analysis,
            'confidence_analysis': confidence_analysis,
            'performance_benchmark': benchmark_results
        }
        
        # 보고서 저장
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path_obj, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 분석 보고서 저장 완료: {save_path_obj}")
        
        return str(save_path_obj) 