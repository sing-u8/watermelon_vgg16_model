"""
Model Evaluator
수박 당도 예측 모델의 종합적인 평가를 위한 모듈
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
from tqdm import tqdm

# 상대 import
import sys
from pathlib import Path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from models.vgg_watermelon import VGGWatermelon, load_model_checkpoint
from training.data_loader import WatermelonDataLoader
from training.metrics import RegressionMetrics, MetricsTracker
from data.watermelon_dataset import WatermelonDataset


class WatermelonEvaluator:
    """
    수박 당도 예측 모델의 종합적인 평가를 위한 클래스
    """
    
    def __init__(self, 
                 model: VGGWatermelon,
                 device: torch.device,
                 save_dir: str = "evaluation_results"):
        """
        WatermelonEvaluator 초기화
        
        Args:
            model (VGGWatermelon): 평가할 모델
            device (torch.device): 디바이스
            save_dir (str): 결과 저장 디렉토리
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 WatermelonEvaluator 초기화 완료")
        print(f"   💾 결과 저장 경로: {self.save_dir}")
        print(f"   🖥️ 디바이스: {self.device}")
    
    def evaluate_model(self, 
                      data_loader: WatermelonDataLoader,
                      split: str = "test") -> Dict[str, Any]:
        """
        모델 평가 수행
        
        Args:
            data_loader (WatermelonDataLoader): 데이터 로더
            split (str): 평가할 데이터 분할 ('train', 'val', 'test')
            
        Returns:
            Dict[str, Any]: 평가 결과
        """
        self.model.eval()
        
        # 데이터 로더 선택
        if split == "train":
            loader = data_loader.train_loader
        elif split == "val":
            loader = data_loader.val_loader
        elif split == "test":
            loader = data_loader.test_loader
        else:
            raise ValueError(f"지원하지 않는 split: {split}")
        
        print(f"🔍 {split.upper()} 데이터 평가 시작")
        
        # 메트릭 초기화
        metrics = RegressionMetrics(self.device)
        
        # 예측 결과 저장
        all_predictions = []
        all_targets = []
        all_losses = []
        
        # 추론 시간 측정
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc=f"Evaluating {split}")):
                # 데이터를 디바이스로 이동
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 추론 시간 측정
                start_time = time.time()
                
                # 순전파
                predictions = self.model(inputs)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # 손실 계산 (MSE 사용)
                loss = nn.MSELoss()(predictions, targets)
                
                # 메트릭 업데이트
                metrics.update(predictions, targets, loss)
                
                # 결과 저장
                pred_np = predictions.detach().cpu().numpy().flatten()
                target_np = targets.detach().cpu().numpy().flatten()
                
                all_predictions.extend(pred_np)
                all_targets.extend(target_np)
                all_losses.append(loss.item())
        
        # 메트릭 계산
        computed_metrics = metrics.compute()
        
        # 추가 분석
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # 성능 분석
        performance_analysis = self._analyze_performance(all_predictions, all_targets)
        
        # 오차 분석
        error_analysis = self._analyze_errors(all_predictions, all_targets)
        
        # 추론 시간 분석
        inference_analysis = {
            'avg_inference_time': np.mean(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'std_inference_time': np.std(inference_times),
            'total_inference_time': np.sum(inference_times),
            'samples_per_second': len(all_predictions) / np.sum(inference_times)
        }
        
        # 종합 결과
        evaluation_results = {
            'split': split,
            'metrics': computed_metrics,
            'performance_analysis': performance_analysis,
            'error_analysis': error_analysis,
            'inference_analysis': inference_analysis,
            'predictions': all_predictions.tolist(),
            'targets': all_targets.tolist(),
            'losses': all_losses
        }
        
        # 결과 출력
        self._print_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def _analyze_performance(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """성능 상세 분석"""
        
        # 당도 범위별 성능 분석
        ranges = {
            'low_sweetness': (8.0, 10.0),    # 낮은 당도
            'medium_sweetness': (10.0, 11.5), # 중간 당도
            'high_sweetness': (11.5, 13.0)    # 높은 당도
        }
        
        range_analysis = {}
        for range_name, (min_val, max_val) in ranges.items():
            mask = (targets >= min_val) & (targets < max_val)
            if np.sum(mask) > 0:
                range_preds = predictions[mask]
                range_targets = targets[mask]
                
                range_mae = np.mean(np.abs(range_preds - range_targets))
                range_rmse = np.sqrt(np.mean((range_preds - range_targets) ** 2))
                range_r2 = 1 - np.sum((range_targets - range_preds) ** 2) / np.sum((range_targets - np.mean(range_targets)) ** 2)
                
                range_analysis[range_name] = {
                    'count': int(np.sum(mask)),
                    'mae': float(range_mae),
                    'rmse': float(range_rmse),
                    'r2_score': float(range_r2),
                    'range': (min_val, max_val)
                }
        
        # 정확도 임계값별 분석
        accuracy_thresholds = [0.1, 0.25, 0.5, 0.75, 1.0]
        accuracy_analysis = {}
        
        errors = np.abs(predictions - targets)
        for threshold in accuracy_thresholds:
            accuracy = np.mean(errors <= threshold) * 100
            accuracy_analysis[f'accuracy_{threshold}'] = float(accuracy)
        
        return {
            'range_analysis': range_analysis,
            'accuracy_analysis': accuracy_analysis
        }
    
    def _analyze_errors(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """오차 패턴 분석"""
        
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        # 기본 오차 통계
        error_stats = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'median_error': float(np.median(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'mean_abs_error': float(np.mean(abs_errors)),
            'median_abs_error': float(np.median(abs_errors))
        }
        
        # 오차 분포 분석
        error_percentiles = [5, 10, 25, 75, 90, 95]
        percentile_analysis = {}
        for p in error_percentiles:
            percentile_analysis[f'percentile_{p}'] = float(np.percentile(abs_errors, p))
        
        # 이상치 분석 (Q3 + 1.5*IQR 이상)
        q1 = np.percentile(abs_errors, 25)
        q3 = np.percentile(abs_errors, 75)
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        
        outliers = abs_errors > outlier_threshold
        outlier_analysis = {
            'outlier_threshold': float(outlier_threshold),
            'num_outliers': int(np.sum(outliers)),
            'outlier_percentage': float(np.mean(outliers) * 100),
            'outlier_indices': np.where(outliers)[0].tolist()
        }
        
        # 과적합/과소적합 분석
        over_predictions = errors > 0  # 과대 예측
        under_predictions = errors < 0
        
        bias_analysis = {
            'over_prediction_rate': float(np.mean(over_predictions) * 100),
            'under_prediction_rate': float(np.mean(under_predictions) * 100),
            'avg_over_prediction': float(np.mean(errors[over_predictions])) if np.sum(over_predictions) > 0 else 0.0,
            'avg_under_prediction': float(np.mean(errors[under_predictions])) if np.sum(under_predictions) > 0 else 0.0
        }
        
        return {
            'error_stats': error_stats,
            'percentile_analysis': percentile_analysis,
            'outlier_analysis': outlier_analysis,
            'bias_analysis': bias_analysis
        }
    
    def _print_evaluation_results(self, results: Dict[str, Any]):
        """평가 결과 출력"""
        split = results['split']
        metrics = results['metrics']
        perf = results['performance_analysis']
        error = results['error_analysis']
        inference = results['inference_analysis']
        
        print(f"\n📊 {split.upper()} 데이터 평가 결과")
        print("=" * 50)
        
        # 기본 메트릭
        print(f"📏 MAE: {metrics['mae']:.3f}")
        print(f"📐 RMSE: {metrics['rmse']:.3f}")
        print(f"🎯 R² Score: {metrics['r2_score']:.3f}")
        print(f"📈 MAPE: {metrics['mape']:.2f}%")
        print(f"🔗 Pearson 상관계수: {metrics['pearson_correlation']:.3f}")
        
        # 당도 정확도
        print(f"✅ 당도 정확도 (±0.5): {metrics['sweetness_accuracy_05']:.1f}%")
        print(f"✅ 당도 정확도 (±1.0): {metrics['sweetness_accuracy_10']:.1f}%")
        
        # 당도 범위별 성능
        print(f"\n📊 당도 범위별 성능:")
        for range_name, range_data in perf['range_analysis'].items():
            print(f"   {range_name}: MAE={range_data['mae']:.3f}, "
                  f"R²={range_data['r2_score']:.3f} (n={range_data['count']})")
        
        # 오차 분석
        print(f"\n📉 오차 분석:")
        print(f"   평균 오차: {error['error_stats']['mean_error']:.3f} ± {error['error_stats']['std_error']:.3f}")
        print(f"   중앙값 오차: {error['error_stats']['median_error']:.3f}")
        print(f"   이상치: {error['outlier_analysis']['num_outliers']}개 ({error['outlier_analysis']['outlier_percentage']:.1f}%)")
        
        # 편향 분석
        print(f"   과대 예측: {error['bias_analysis']['over_prediction_rate']:.1f}%")
        print(f"   과소 예측: {error['bias_analysis']['under_prediction_rate']:.1f}%")
        
        # 추론 성능
        print(f"\n⚡ 추론 성능:")
        print(f"   평균 추론 시간: {inference['avg_inference_time']*1000:.2f}ms")
        print(f"   초당 샘플 수: {inference['samples_per_second']:.1f}")
    
    def compare_models(self, 
                      model_paths: List[str],
                      model_names: List[str],
                      data_loader: WatermelonDataLoader,
                      split: str = "test") -> Dict[str, Any]:
        """
        여러 모델 성능 비교
        
        Args:
            model_paths (List[str]): 모델 체크포인트 경로들
            model_names (List[str]): 모델 이름들
            data_loader (WatermelonDataLoader): 데이터 로더
            split (str): 평가할 데이터 분할
            
        Returns:
            Dict[str, Any]: 모델 비교 결과
        """
        print(f"🔍 모델 성능 비교 시작 ({len(model_paths)}개 모델)")
        
        comparison_results = {}
        
        for model_path, model_name in zip(model_paths, model_names):
            print(f"\n📊 {model_name} 평가 중...")
            
            # 모델 로드
            model = load_model_checkpoint(model_path)
            model.to(self.device)
            
            # 평가 수행
            evaluator = WatermelonEvaluator(model, self.device, 
                                          str(self.save_dir / f"comparison_{model_name}"))
            results = evaluator.evaluate_model(data_loader, split)
            
            comparison_results[model_name] = results
        
        # 비교 분석
        comparison_analysis = self._analyze_model_comparison(comparison_results)
        
        # 비교 결과 저장
        self._save_comparison_results(comparison_results, comparison_analysis)
        
        return {
            'individual_results': comparison_results,
            'comparison_analysis': comparison_analysis
        }
    
    def _analyze_model_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """모델 비교 분석"""
        
        # 메트릭별 순위
        metrics_comparison = {}
        metric_names = ['mae', 'rmse', 'r2_score', 'mape', 'sweetness_accuracy_05', 'sweetness_accuracy_10']
        
        for metric in metric_names:
            metric_values = {}
            for model_name, result in results.items():
                metric_values[model_name] = result['metrics'][metric]
            
            # 정렬 (MAE, RMSE, MAPE는 낮을수록 좋음, 나머지는 높을수록 좋음)
            reverse = metric not in ['mae', 'rmse', 'mape']
            sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=reverse)
            
            metrics_comparison[metric] = {
                'ranking': [(name, value) for name, value in sorted_models],
                'best_model': sorted_models[0][0],
                'best_value': sorted_models[0][1]
            }
        
        # 종합 순위 (여러 메트릭의 순위 평균)
        model_ranks = {name: [] for name in results.keys()}
        
        for metric, comparison in metrics_comparison.items():
            for rank, (model_name, _) in enumerate(comparison['ranking']):
                model_ranks[model_name].append(rank + 1)
        
        overall_ranking = []
        for model_name, ranks in model_ranks.items():
            avg_rank = np.mean(ranks)
            overall_ranking.append((model_name, avg_rank))
        
        overall_ranking.sort(key=lambda x: x[1])
        
        return {
            'metrics_comparison': metrics_comparison,
            'overall_ranking': overall_ranking,
            'best_overall_model': overall_ranking[0][0]
        }
    
    def _save_comparison_results(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """모델 비교 결과 저장"""
        
        # JSON 저장
        comparison_data = {
            'comparison_results': results,
            'comparison_analysis': analysis
        }
        
        save_path = self.save_dir / "model_comparison.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 모델 비교 결과 저장됨: {save_path}")
    
    def generate_evaluation_report(self, 
                                  evaluation_results: Dict[str, Any],
                                  include_plots: bool = True) -> str:
        """
        종합 평가 보고서 생성
        
        Args:
            evaluation_results (Dict): 평가 결과
            include_plots (bool): 플롯 포함 여부
            
        Returns:
            str: 생성된 보고서 경로
        """
        print("📝 평가 보고서 생성 중...")
        
        # 보고서 내용 생성
        report_content = self._generate_report_content(evaluation_results)
        
        # 보고서 저장
        report_path = self.save_dir / "evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 플롯 생성
        if include_plots:
            self._generate_evaluation_plots(evaluation_results)
        
        print(f"📄 평가 보고서 생성 완료: {report_path}")
        return str(report_path)
    
    def _generate_report_content(self, results: Dict[str, Any]) -> str:
        """보고서 내용 생성"""
        
        split = results['split']
        metrics = results['metrics']
        perf = results['performance_analysis']
        error = results['error_analysis']
        inference = results['inference_analysis']
        
        report = f"""# 🍉 Watermelon Sweetness Prediction - Evaluation Report

## 📊 Overview
- **Dataset Split**: {split.upper()}
- **Total Samples**: {metrics['num_samples']:,}
- **Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Model Performance

### Core Metrics
| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | {metrics['mae']:.3f} |
| RMSE (Root Mean Square Error) | {metrics['rmse']:.3f} |
| R² Score | {metrics['r2_score']:.3f} |
| MAPE (Mean Absolute Percentage Error) | {metrics['mape']:.2f}% |
| Pearson Correlation | {metrics['pearson_correlation']:.3f} |

### Sweetness Accuracy
| Tolerance | Accuracy |
|-----------|----------|
| ±0.5 Brix | {metrics['sweetness_accuracy_05']:.1f}% |
| ±1.0 Brix | {metrics['sweetness_accuracy_10']:.1f}% |

## 📈 Performance by Sweetness Range

"""
        
        # 당도 범위별 성능 추가
        for range_name, range_data in perf['range_analysis'].items():
            range_display = range_name.replace('_', ' ').title()
            report += f"### {range_display} ({range_data['range'][0]}-{range_data['range'][1]} Brix)\n"
            report += f"- **Samples**: {range_data['count']}\n"
            report += f"- **MAE**: {range_data['mae']:.3f}\n"
            report += f"- **RMSE**: {range_data['rmse']:.3f}\n"
            report += f"- **R² Score**: {range_data['r2_score']:.3f}\n\n"
        
        # 오차 분석 추가
        report += f"""## 📉 Error Analysis

### Error Statistics
- **Mean Error**: {error['error_stats']['mean_error']:.3f} ± {error['error_stats']['std_error']:.3f}
- **Median Error**: {error['error_stats']['median_error']:.3f}
- **Max Absolute Error**: {error['error_stats']['max_error']:.3f}

### Prediction Bias
- **Over-prediction Rate**: {error['bias_analysis']['over_prediction_rate']:.1f}%
- **Under-prediction Rate**: {error['bias_analysis']['under_prediction_rate']:.1f}%

### Outliers
- **Outlier Threshold**: {error['outlier_analysis']['outlier_threshold']:.3f}
- **Number of Outliers**: {error['outlier_analysis']['num_outliers']} ({error['outlier_analysis']['outlier_percentage']:.1f}%)

## ⚡ Inference Performance

| Metric | Value |
|--------|-------|
| Average Inference Time | {inference['avg_inference_time']*1000:.2f} ms |
| Samples per Second | {inference['samples_per_second']:.1f} |
| Total Inference Time | {inference['total_inference_time']:.2f} s |

## 🎯 Model Assessment

### Strengths
"""
        
        # 강점 분석
        strengths = []
        if metrics['r2_score'] > 0.8:
            strengths.append("- Excellent correlation between predictions and actual values (R² > 0.8)")
        if metrics['sweetness_accuracy_05'] > 70:
            strengths.append("- High accuracy within ±0.5 Brix tolerance")
        if inference['avg_inference_time'] < 0.1:
            strengths.append("- Fast inference suitable for real-time applications")
        
        for strength in strengths:
            report += f"{strength}\n"
        
        # 개선 영역
        report += "\n### Areas for Improvement\n"
        improvements = []
        if metrics['mae'] > 0.5:
            improvements.append("- Consider reducing MAE for better accuracy")
        if error['bias_analysis']['over_prediction_rate'] > 60 or error['bias_analysis']['under_prediction_rate'] > 60:
            improvements.append("- Address prediction bias for more balanced predictions")
        if error['outlier_analysis']['outlier_percentage'] > 10:
            improvements.append("- Investigate and reduce outlier predictions")
        
        for improvement in improvements:
            report += f"{improvement}\n"
        
        report += f"""
## 📊 Visualizations

The following plots are available in the evaluation results:
- Prediction vs Target Scatter Plot
- Error Distribution Histogram
- Residual Plot
- Performance by Sweetness Range

---
*Report generated by WatermelonEvaluator*
"""
        
        return report
    
    def _generate_evaluation_plots(self, results: Dict[str, Any]):
        """평가 플롯 생성"""
        
        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        
        # 플롯 스타일 설정
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Prediction vs Target 산점도
        plt.figure(figsize=(10, 8))
        plt.scatter(targets, predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # 완벽한 예측 라인
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # 허용 오차 영역
        plt.fill_between([min_val, max_val], [min_val-0.5, max_val-0.5], 
                        [min_val+0.5, max_val+0.5], alpha=0.2, color='green', 
                        label='±0.5 Brix tolerance')
        
        plt.xlabel('Actual Sweetness (Brix)', fontsize=12)
        plt.ylabel('Predicted Sweetness (Brix)', fontsize=12)
        plt.title('🍉 Watermelon Sweetness Prediction Results', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "prediction_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 오차 분포 히스토그램
        errors = predictions - targets
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(float(np.mean(errors)), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
        plt.xlabel('Prediction Error (Brix)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 절대 오차 분포
        plt.subplot(2, 2, 2)
        abs_errors = np.abs(errors)
        plt.hist(abs_errors, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.axvline(float(np.mean(abs_errors)), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(abs_errors):.3f}')
        plt.xlabel('Absolute Error (Brix)')
        plt.ylabel('Frequency')
        plt.title('Absolute Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 잔차 플롯
        plt.subplot(2, 2, 3)
        plt.scatter(predictions, errors, alpha=0.6, s=30)
        plt.axhline(0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Sweetness (Brix)')
        plt.ylabel('Residuals (Brix)')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # 5. Q-Q 플롯
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normal Distribution)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "error_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 평가 플롯 저장됨: {self.save_dir}")


# 사용 편의를 위한 헬퍼 함수들
def create_evaluator(model_path: str, device: str = "auto") -> WatermelonEvaluator:
    """
    평가기를 쉽게 생성하는 헬퍼 함수
    
    Args:
        model_path (str): 모델 체크포인트 경로
        device (str): 디바이스 ('auto', 'cpu', 'cuda')
        
    Returns:
        WatermelonEvaluator: 초기화된 평가기
    """
    # 디바이스 설정
    if device == "auto":
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        torch_device = torch.device(device)
    
    # 모델 로드
    model = load_model_checkpoint(model_path)
    model.to(torch_device)
    
    # 평가기 생성
    evaluator = WatermelonEvaluator(model, torch_device)
    
    return evaluator


def quick_evaluation(model_path: str, data_path: str, split: str = "test") -> Dict[str, Any]:
    """
    빠른 모델 평가를 위한 헬퍼 함수
    
    Args:
        model_path (str): 모델 체크포인트 경로
        data_path (str): 데이터셋 경로
        split (str): 평가할 분할
        
    Returns:
        Dict[str, Any]: 평가 결과
    """
    # 평가기 생성
    evaluator = create_evaluator(model_path)
    
    # 데이터 로더 생성
    from training.data_loader import create_data_loaders
    data_loader = create_data_loaders(
        data_path=data_path,
        batch_size=32,
        num_workers=4
    )
    
    # 평가 수행
    results = evaluator.evaluate_model(data_loader, split)
    
    # 보고서 생성
    evaluator.generate_evaluation_report(results)
    
    return results


if __name__ == "__main__":
    # 평가기 테스트
    print("🧪 평가기 모듈 테스트")
    
    try:
        # 더미 모델 생성
        from models.vgg_watermelon import create_vgg_watermelon
        model = create_vgg_watermelon()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 평가기 생성
        evaluator = WatermelonEvaluator(model, device, "test_evaluation")
        
        print("✅ 평가기 테스트 완료")
        
    except Exception as e:
        print(f"❌ 평가기 테스트 실패: {e}")
        import traceback
        traceback.print_exc() 