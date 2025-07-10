"""
Metrics Module
수박 당도 예측 모델 평가를 위한 메트릭 모듈
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path


class RegressionMetrics:
    """
    회귀 모델 평가를 위한 메트릭 클래스
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        RegressionMetrics 초기화
        
        Args:
            device (torch.device, optional): 계산에 사용할 디바이스
        """
        self.device = device or torch.device('cpu')
        self.reset()
    
    def reset(self):
        """메트릭 초기화"""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, 
               predictions: torch.Tensor, 
               targets: torch.Tensor, 
               loss: Optional[torch.Tensor] = None):
        """
        배치 결과로 메트릭 업데이트
        
        Args:
            predictions (torch.Tensor): 모델 예측값
            targets (torch.Tensor): 실제 타겟값
            loss (torch.Tensor, optional): 손실값
        """
        # CPU로 이동 및 numpy 변환
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        self.predictions.extend(pred_np)
        self.targets.extend(target_np)
        
        if loss is not None:
            loss_val = loss.detach().cpu().item()
            self.losses.append(loss_val)
    
    def compute(self) -> Dict[str, float]:
        """
        모든 메트릭 계산
        
        Returns:
            Dict[str, float]: 계산된 메트릭들
        """
        if len(self.predictions) == 0:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # 기본 회귀 메트릭
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # 추가 메트릭
        mape = self._calculate_mape(targets, predictions)
        pearson_corr = self._calculate_pearson_correlation(targets, predictions)
        sweetness_accuracy = self._calculate_sweetness_accuracy(targets, predictions)
        
        # 오차 통계
        errors = predictions - targets
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(np.abs(errors))
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape,
            'pearson_correlation': pearson_corr,
            'sweetness_accuracy_05': sweetness_accuracy[0],  # ±0.5 Brix
            'sweetness_accuracy_10': sweetness_accuracy[1],  # ±1.0 Brix
            'mean_error': mean_error,
            'std_error': std_error,
            'max_error': max_error,
            'num_samples': len(predictions)
        }
        
        # 손실 평균 추가
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
        
        return metrics
    
    def _calculate_mape(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """평균 절대 백분율 오차 (MAPE) 계산"""
        # 0으로 나누기 방지
        epsilon = 1e-8
        return float(np.mean(np.abs((targets - predictions) / (targets + epsilon))) * 100)
    
    def _calculate_pearson_correlation(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """피어슨 상관계수 계산"""
        correlation_matrix = np.corrcoef(targets, predictions)
        return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
    
    def _calculate_sweetness_accuracy(self, targets: np.ndarray, predictions: np.ndarray) -> Tuple[float, float]:
        """당도 예측 정확도 계산 (±0.5, ±1.0 Brix 내 정확도)"""
        errors = np.abs(targets - predictions)
        
        accuracy_05 = float(np.mean(errors <= 0.5) * 100)  # ±0.5 Brix 내 정확도
        accuracy_10 = float(np.mean(errors <= 1.0) * 100)  # ±1.0 Brix 내 정확도
        
        return accuracy_05, accuracy_10
    
    def print_metrics(self):
        """메트릭 출력"""
        metrics = self.compute()
        
        if not metrics:
            print("📊 메트릭: 데이터 없음")
            return
        
        print(f"📊 회귀 메트릭 결과 (샘플 수: {metrics['num_samples']:,})")
        print(f"   📏 MAE: {metrics['mae']:.3f}")
        print(f"   📐 MSE: {metrics['mse']:.3f}")
        print(f"   📊 RMSE: {metrics['rmse']:.3f}")
        print(f"   🎯 R² Score: {metrics['r2_score']:.3f}")
        print(f"   📈 MAPE: {metrics['mape']:.2f}%")
        print(f"   🔗 Pearson 상관계수: {metrics['pearson_correlation']:.3f}")
        print(f"   ✅ 당도 정확도 (±0.5): {metrics['sweetness_accuracy_05']:.1f}%")
        print(f"   ✅ 당도 정확도 (±1.0): {metrics['sweetness_accuracy_10']:.1f}%")
        print(f"   📊 평균 오차: {metrics['mean_error']:.3f} ± {metrics['std_error']:.3f}")
        print(f"   🔴 최대 오차: {metrics['max_error']:.3f}")
        
        if 'avg_loss' in metrics:
            print(f"   💔 평균 손실: {metrics['avg_loss']:.4f}")


class MetricsTracker:
    """
    훈련 과정에서 메트릭을 추적하는 클래스
    """
    
    def __init__(self):
        """MetricsTracker 초기화"""
        self.train_metrics = []
        self.val_metrics = []
        self.epoch_losses = []
    
    def add_epoch_metrics(self, 
                         train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float],
                         epoch_loss: Optional[float] = None):
        """
        에포크 메트릭 추가
        
        Args:
            train_metrics (Dict): 훈련 메트릭
            val_metrics (Dict): 검증 메트릭
            epoch_loss (float, optional): 에포크 손실
        """
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)
        
        if epoch_loss is not None:
            self.epoch_losses.append(epoch_loss)
    
    def get_best_epoch(self, metric_name: str = 'val_mae', minimize: bool = True) -> int:
        """
        최적 에포크 찾기
        
        Args:
            metric_name (str): 기준 메트릭 이름
            minimize (bool): 최소화 여부 (True: 최소값, False: 최대값)
            
        Returns:
            int: 최적 에포크 번호
        """
        if not self.val_metrics:
            return -1
        
        # 검증 메트릭에서 해당 메트릭 추출
        metric_key = metric_name.replace('val_', '')
        values = [metrics.get(metric_key, float('inf') if minimize else float('-inf')) 
                 for metrics in self.val_metrics]
        
        if minimize:
            return int(np.argmin(values))
        else:
            return int(np.argmax(values))
    
    def plot_training_curves(self, 
                           save_path: Optional[str] = None,
                           show: bool = True):
        """
        훈련 곡선 플롯
        
        Args:
            save_path (str, optional): 저장 경로
            show (bool): 플롯 표시 여부
        """
        if not self.train_metrics or not self.val_metrics:
            print("⚠️ 플롯할 메트릭 데이터가 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🍉 Watermelon Sweetness Prediction - Training Curves', fontsize=16)
        
        epochs = range(1, len(self.train_metrics) + 1)
        
        # 1. MAE 곡선
        train_mae = [m.get('mae', 0) for m in self.train_metrics]
        val_mae = [m.get('mae', 0) for m in self.val_metrics]
        
        axes[0, 0].plot(epochs, train_mae, 'b-', label='Train MAE', linewidth=2)
        axes[0, 0].plot(epochs, val_mae, 'r-', label='Val MAE', linewidth=2)
        axes[0, 0].set_title('Mean Absolute Error (MAE)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. R² Score 곡선
        train_r2 = [m.get('r2_score', 0) for m in self.train_metrics]
        val_r2 = [m.get('r2_score', 0) for m in self.val_metrics]
        
        axes[0, 1].plot(epochs, train_r2, 'b-', label='Train R²', linewidth=2)
        axes[0, 1].plot(epochs, val_r2, 'r-', label='Val R²', linewidth=2)
        axes[0, 1].set_title('R² Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. RMSE 곡선
        train_rmse = [m.get('rmse', 0) for m in self.train_metrics]
        val_rmse = [m.get('rmse', 0) for m in self.val_metrics]
        
        axes[1, 0].plot(epochs, train_rmse, 'b-', label='Train RMSE', linewidth=2)
        axes[1, 0].plot(epochs, val_rmse, 'r-', label='Val RMSE', linewidth=2)
        axes[1, 0].set_title('Root Mean Square Error (RMSE)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 당도 정확도 곡선
        train_acc = [m.get('sweetness_accuracy_05', 0) for m in self.train_metrics]
        val_acc = [m.get('sweetness_accuracy_05', 0) for m in self.val_metrics]
        
        axes[1, 1].plot(epochs, train_acc, 'b-', label='Train Acc (±0.5)', linewidth=2)
        axes[1, 1].plot(epochs, val_acc, 'r-', label='Val Acc (±0.5)', linewidth=2)
        axes[1, 1].set_title('Sweetness Accuracy (±0.5 Brix)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 훈련 곡선 저장됨: {save_path}")
        
        # 표시
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_prediction_scatter(self, 
                              predictions: np.ndarray,
                              targets: np.ndarray,
                              title: str = "Prediction vs Target",
                              save_path: Optional[str] = None,
                              show: bool = True):
        """
        예측값 vs 실제값 산점도 플롯
        
        Args:
            predictions (np.ndarray): 예측값
            targets (np.ndarray): 실제값
            title (str): 플롯 제목
            save_path (str, optional): 저장 경로
            show (bool): 플롯 표시 여부
        """
        plt.figure(figsize=(10, 8))
        
        # 산점도
        plt.scatter(targets, predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # 완벽한 예측 라인 (y=x)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # ±0.5, ±1.0 Brix 허용 오차 영역
        plt.fill_between([min_val, max_val], [min_val-0.5, max_val-0.5], 
                        [min_val+0.5, max_val+0.5], alpha=0.2, color='green', 
                        label='±0.5 Brix tolerance')
        plt.fill_between([min_val, max_val], [min_val-1.0, max_val-1.0], 
                        [min_val+1.0, max_val+1.0], alpha=0.1, color='blue', 
                        label='±1.0 Brix tolerance')
        
        # 메트릭 계산
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        plt.xlabel('Actual Sweetness (Brix)', fontsize=12)
        plt.ylabel('Predicted Sweetness (Brix)', fontsize=12)
        plt.title(f'{title}\nMAE: {mae:.3f}, R²: {r2:.3f}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 축 범위 설정
        plt.xlim(min_val - 0.5, max_val + 0.5)
        plt.ylim(min_val - 0.5, max_val + 0.5)
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 산점도 저장됨: {save_path}")
        
        # 표시
        if show:
            plt.show()
        else:
            plt.close()
    
    def _convert_numpy_to_python(self, obj):
        """
        NumPy 타입을 Python 기본 타입으로 변환하는 헬퍼 함수
        
        Args:
            obj: 변환할 객체
            
        Returns:
            변환된 객체 (JSON 직렬화 가능)
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_python(item) for item in obj]
        else:
            return obj
    
    def save_metrics_summary(self, save_path: str):
        """
        메트릭 요약 저장
        
        Args:
            save_path (str): 저장 경로
        """
        if not self.train_metrics or not self.val_metrics:
            print("⚠️ 저장할 메트릭 데이터가 없습니다.")
            return
        
        # 최종 메트릭 추출
        final_train = self.train_metrics[-1]
        final_val = self.val_metrics[-1]
        
        # 최적 에포크 찾기
        best_mae_epoch = self.get_best_epoch('mae', minimize=True)
        best_r2_epoch = self.get_best_epoch('r2_score', minimize=False)
        
        summary = {
            'training_summary': {
                'total_epochs': len(self.train_metrics),
                'best_mae_epoch': best_mae_epoch + 1,
                'best_r2_epoch': best_r2_epoch + 1
            },
            'final_metrics': {
                'train': final_train,
                'validation': final_val
            },
            'best_metrics': {
                'best_mae': {
                    'epoch': best_mae_epoch + 1,
                    'train': self.train_metrics[best_mae_epoch],
                    'val': self.val_metrics[best_mae_epoch]
                },
                'best_r2': {
                    'epoch': best_r2_epoch + 1,
                    'train': self.train_metrics[best_r2_epoch],
                    'val': self.val_metrics[best_r2_epoch]
                }
            }
        }
        
        # NumPy 타입을 Python 타입으로 변환
        summary = self._convert_numpy_to_python(summary)
        
        # JSON으로 저장
        import json
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 메트릭 요약 저장됨: {save_path}")


if __name__ == "__main__":
    # 메트릭 테스트
    print("🧪 메트릭 모듈 테스트")
    
    # 테스트 데이터 생성
    np.random.seed(42)
    targets = np.random.uniform(8.0, 13.0, 100)  # 실제 당도값
    noise = np.random.normal(0, 0.5, 100)  # 노이즈
    predictions = targets + noise  # 노이즈가 있는 예측값
    
    # 메트릭 계산
    metrics = RegressionMetrics()
    
    # 배치별로 업데이트 시뮬레이션
    batch_size = 10
    for i in range(0, len(targets), batch_size):
        batch_targets = torch.tensor(targets[i:i+batch_size], dtype=torch.float32)
        batch_predictions = torch.tensor(predictions[i:i+batch_size], dtype=torch.float32)
        
        metrics.update(batch_predictions, batch_targets)
    
    # 결과 출력
    metrics.print_metrics()
    
    # 메트릭 추적기 테스트
    tracker = MetricsTracker()
    
    # 가상의 훈련 과정 시뮬레이션
    for epoch in range(5):
        train_m = {
            'mae': 1.0 - epoch * 0.1,
            'r2_score': 0.5 + epoch * 0.08,
            'rmse': 1.5 - epoch * 0.15
        }
        val_m = {
            'mae': 1.2 - epoch * 0.08,
            'r2_score': 0.4 + epoch * 0.09,
            'rmse': 1.8 - epoch * 0.12
        }
        tracker.add_epoch_metrics(train_m, val_m)
    
    print(f"\n🎯 최적 에포크 (MAE 기준): {tracker.get_best_epoch('mae') + 1}")
    print(f"🎯 최적 에포크 (R² 기준): {tracker.get_best_epoch('r2_score', minimize=False) + 1}")
    
    print("✅ 메트릭 모듈 테스트 완료") 