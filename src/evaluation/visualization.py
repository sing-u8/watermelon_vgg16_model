"""
Visualization Module
모델 평가 결과 시각화를 위한 모듈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (선택적)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def create_evaluation_plots(evaluation_results: Dict[str, Any], 
                          save_dir: str = "evaluation_plots",
                          show_plots: bool = False) -> List[str]:
    """
    평가 결과에서 다양한 플롯 생성
    
    Args:
        evaluation_results (Dict): 평가 결과 딕셔너리
        save_dir (str): 플롯 저장 디렉토리
        show_plots (bool): 플롯 표시 여부
        
    Returns:
        List[str]: 생성된 플롯 파일 경로들
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = np.array(evaluation_results['predictions'])
    targets = np.array(evaluation_results['targets'])
    
    generated_plots = []
    
    # 스타일 설정
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Prediction vs Target 산점도
    plot_path = save_dir / "prediction_vs_target.png"
    create_prediction_scatter_plot(predictions, targets, str(plot_path), show_plots)
    generated_plots.append(str(plot_path))
    
    # 2. 오차 분석 플롯
    plot_path = save_dir / "error_analysis.png"
    create_error_analysis_plots(predictions, targets, str(plot_path), show_plots)
    generated_plots.append(str(plot_path))
    
    # 3. 성능 메트릭 비교
    plot_path = save_dir / "metrics_comparison.png"
    create_metrics_comparison_plot(evaluation_results['metrics'], str(plot_path), show_plots)
    generated_plots.append(str(plot_path))
    
    # 4. 당도 범위별 성능
    if 'performance_analysis' in evaluation_results:
        plot_path = save_dir / "sweetness_range_performance.png"
        create_sweetness_range_plot(evaluation_results['performance_analysis'], str(plot_path), show_plots)
        generated_plots.append(str(plot_path))
    
    # 5. 예측 분포
    plot_path = save_dir / "prediction_distribution.png"
    create_prediction_distribution_plot(predictions, targets, str(plot_path), show_plots)
    generated_plots.append(str(plot_path))
    
    print(f"📊 {len(generated_plots)}개 평가 플롯 생성 완료: {save_dir}")
    
    return generated_plots


def create_prediction_scatter_plot(predictions: np.ndarray, 
                                 targets: np.ndarray,
                                 save_path: str,
                                 show_plot: bool = False):
    """예측 vs 실제값 산점도 생성"""
    
    plt.figure(figsize=(10, 8))
    
    # 산점도
    plt.scatter(targets, predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='steelblue')
    
    # 완벽한 예측 라인
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 허용 오차 영역들
    plt.fill_between([min_val, max_val], [min_val-0.5, max_val-0.5], 
                    [min_val+0.5, max_val+0.5], alpha=0.2, color='green', 
                    label='±0.5 Brix tolerance')
    plt.fill_between([min_val, max_val], [min_val-1.0, max_val-1.0], 
                    [min_val+1.0, max_val+1.0], alpha=0.1, color='orange', 
                    label='±1.0 Brix tolerance')
    
    # 메트릭 계산 및 표시
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    
    plt.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Actual Sweetness (Brix)', fontsize=12)
    plt.ylabel('Predicted Sweetness (Brix)', fontsize=12)
    plt.title('🍉 Watermelon Sweetness Prediction Results', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 축 범위 동일하게 설정
    plt.axis('equal')
    plt.xlim(min_val - 0.5, max_val + 0.5)
    plt.ylim(min_val - 0.5, max_val + 0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_error_analysis_plots(predictions: np.ndarray, 
                              targets: np.ndarray,
                              save_path: str,
                              show_plot: bool = False):
    """오차 분석 플롯들 생성"""
    
    errors = predictions - targets
    abs_errors = np.abs(errors)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 오차 분포 히스토그램
    axes[0, 0].hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(errors):.3f}')
    axes[0, 0].axvline(0, color='green', linestyle='-', linewidth=1, alpha=0.7, label='Perfect')
    axes[0, 0].set_xlabel('Prediction Error (Brix)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 절대 오차 분포
    axes[0, 1].hist(abs_errors, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].axvline(np.mean(abs_errors), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(abs_errors):.3f}')
    axes[0, 1].set_xlabel('Absolute Error (Brix)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Absolute Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 잔차 플롯
    axes[1, 0].scatter(predictions, errors, alpha=0.6, s=30, color='purple')
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Sweetness (Brix)')
    axes[1, 0].set_ylabel('Residuals (Brix)')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q 플롯
    try:
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 1].grid(True, alpha=0.3)
    except ImportError:
        # scipy가 없는 경우 대안
        sorted_errors = np.sort(errors)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_errors)))
        axes[1, 1].scatter(theoretical_quantiles, sorted_errors, alpha=0.6)
        axes[1, 1].plot(theoretical_quantiles, theoretical_quantiles, 'r--', linewidth=2)
        axes[1, 1].set_xlabel('Theoretical Quantiles')
        axes[1, 1].set_ylabel('Sample Quantiles')
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_metrics_comparison_plot(metrics: Dict[str, float],
                                 save_path: str,
                                 show_plot: bool = False):
    """메트릭 비교 차트 생성"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 주요 메트릭 바 차트
    main_metrics = ['mae', 'rmse', 'r2_score', 'mape']
    main_values = [metrics.get(metric, 0) for metric in main_metrics]
    main_labels = ['MAE', 'RMSE', 'R² Score', 'MAPE (%)']
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
    bars = axes[0].bar(main_labels, main_values, color=colors, alpha=0.7, edgecolor='black')
    
    # 값 표시
    for bar, value in zip(bars, main_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    axes[0].set_title('Core Performance Metrics', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Metric Value')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. 당도 정확도 파이 차트
    accuracy_05 = metrics.get('sweetness_accuracy_05', 0)
    accuracy_10 = metrics.get('sweetness_accuracy_10', 0)
    
    # ±0.5 정확도 파이 차트
    pie_data = [accuracy_05, 100 - accuracy_05]
    pie_labels = [f'Within ±0.5 Brix\n({accuracy_05:.1f}%)', 
                  f'Outside ±0.5 Brix\n({100-accuracy_05:.1f}%)']
    
    axes[1].pie(pie_data, labels=pie_labels, autopct='', colors=['lightgreen', 'lightcoral'],
               startangle=90, explode=(0.05, 0))
    axes[1].set_title(f'Sweetness Accuracy (±0.5 Brix)\nTotal: {accuracy_05:.1f}%', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_sweetness_range_plot(performance_analysis: Dict[str, Any],
                              save_path: str,
                              show_plot: bool = False):
    """당도 범위별 성능 분석 플롯 생성"""
    
    range_analysis = performance_analysis.get('range_analysis', {})
    
    if not range_analysis:
        print("Warning: 당도 범위별 분석 데이터가 없습니다.")
        return
    
    # 데이터 준비
    range_names = []
    mae_values = []
    rmse_values = []
    r2_values = []
    sample_counts = []
    
    for range_name, data in range_analysis.items():
        range_names.append(range_name.replace('_', ' ').title())
        mae_values.append(data['mae'])
        rmse_values.append(data['rmse'])
        r2_values.append(data['r2_score'])
        sample_counts.append(data['count'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    x_pos = np.arange(len(range_names))
    
    # 1. MAE 비교
    bars1 = axes[0, 0].bar(x_pos, mae_values, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('MAE by Sweetness Range', fontweight='bold')
    axes[0, 0].set_ylabel('MAE (Brix)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(range_names, rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar, value in zip(bars1, mae_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # 2. RMSE 비교
    bars2 = axes[0, 1].bar(x_pos, rmse_values, color='lightcoral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('RMSE by Sweetness Range', fontweight='bold')
    axes[0, 1].set_ylabel('RMSE (Brix)')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(range_names, rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars2, rmse_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # 3. R² Score 비교
    bars3 = axes[1, 0].bar(x_pos, r2_values, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('R² Score by Sweetness Range', fontweight='bold')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(range_names, rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars3, r2_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # 4. 샘플 수 비교
    bars4 = axes[1, 1].bar(x_pos, sample_counts, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Sample Count by Sweetness Range', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(range_names, rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars4, sample_counts):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_prediction_distribution_plot(predictions: np.ndarray,
                                      targets: np.ndarray,
                                      save_path: str,
                                      show_plot: bool = False):
    """예측값과 실제값의 분포 비교 플롯 생성"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 히스토그램 비교
    axes[0, 0].hist(targets, bins=20, alpha=0.7, label='Actual', color='skyblue', edgecolor='black')
    axes[0, 0].hist(predictions, bins=20, alpha=0.7, label='Predicted', color='lightcoral', edgecolor='black')
    axes[0, 0].set_xlabel('Sweetness (Brix)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 박스 플롯 비교
    box_data = [targets, predictions]
    box_labels = ['Actual', 'Predicted']
    
    bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    axes[0, 1].set_ylabel('Sweetness (Brix)')
    axes[0, 1].set_title('Box Plot Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 밀도 플롯
    try:
        from scipy.stats import gaussian_kde
        
        # 실제값 밀도
        density_actual = gaussian_kde(targets)
        xs_actual = np.linspace(targets.min(), targets.max(), 200)
        axes[1, 0].plot(xs_actual, density_actual(xs_actual), label='Actual', color='blue', linewidth=2)
        
        # 예측값 밀도
        density_pred = gaussian_kde(predictions)
        xs_pred = np.linspace(predictions.min(), predictions.max(), 200)
        axes[1, 0].plot(xs_pred, density_pred(xs_pred), label='Predicted', color='red', linewidth=2)
        
        axes[1, 0].set_xlabel('Sweetness (Brix)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Density Plot Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
    except ImportError:
        # scipy가 없는 경우 히스토그램으로 대체
        axes[1, 0].hist(targets, bins=20, alpha=0.5, density=True, label='Actual', color='blue')
        axes[1, 0].hist(predictions, bins=20, alpha=0.5, density=True, label='Predicted', color='red')
        axes[1, 0].set_xlabel('Sweetness (Brix)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Normalized Histogram Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 통계 요약 테이블
    stats_data = {
        'Metric': ['Mean', 'Std', 'Min', 'Max', 'Median'],
        'Actual': [
            f'{np.mean(targets):.3f}',
            f'{np.std(targets):.3f}',
            f'{np.min(targets):.3f}',
            f'{np.max(targets):.3f}',
            f'{np.median(targets):.3f}'
        ],
        'Predicted': [
            f'{np.mean(predictions):.3f}',
            f'{np.std(predictions):.3f}',
            f'{np.min(predictions):.3f}',
            f'{np.max(predictions):.3f}',
            f'{np.median(predictions):.3f}'
        ]
    }
    
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table = axes[1, 1].table(cellText=[[stats_data['Metric'][i], 
                                       stats_data['Actual'][i], 
                                       stats_data['Predicted'][i]] for i in range(5)],
                            colLabels=['Metric', 'Actual', 'Predicted'],
                            cellLoc='center',
                            loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 헤더 스타일링
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Statistical Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_model_comparison_plot(comparison_results: Dict[str, Any],
                               save_path: str,
                               show_plot: bool = False):
    """여러 모델 성능 비교 플롯 생성"""
    
    if 'comparison_analysis' not in comparison_results:
        print("Warning: 모델 비교 분석 데이터가 없습니다.")
        return
    
    metrics_comparison = comparison_results['comparison_analysis']['metrics_comparison']
    
    # 메트릭별 모델 성능 데이터 준비
    model_names = []
    mae_values = []
    rmse_values = []
    r2_values = []
    
    # 첫 번째 메트릭에서 모델 이름들 추출
    first_metric = next(iter(metrics_comparison.values()))
    for model_name, _ in first_metric['ranking']:
        model_names.append(model_name)
        
        # 각 메트릭 값 추출
        mae_values.append(metrics_comparison['mae']['ranking'][model_names.index(model_name)][1])
        rmse_values.append(metrics_comparison['rmse']['ranking'][model_names.index(model_name)][1])
        r2_values.append(metrics_comparison['r2_score']['ranking'][model_names.index(model_name)][1])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x_pos = np.arange(len(model_names))
    
    # 1. MAE 비교
    bars1 = axes[0].bar(x_pos, mae_values, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0].set_title('MAE Comparison', fontweight='bold')
    axes[0].set_ylabel('MAE (Brix)')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_names, rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 최고 성능 모델 강조
    best_idx = np.argmin(mae_values)
    bars1[best_idx].set_color('gold')
    
    # 2. RMSE 비교
    bars2 = axes[1].bar(x_pos, rmse_values, color='lightcoral', alpha=0.7, edgecolor='black')
    axes[1].set_title('RMSE Comparison', fontweight='bold')
    axes[1].set_ylabel('RMSE (Brix)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(model_names, rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    best_idx = np.argmin(rmse_values)
    bars2[best_idx].set_color('gold')
    
    # 3. R² Score 비교
    bars3 = axes[2].bar(x_pos, r2_values, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[2].set_title('R² Score Comparison', fontweight='bold')
    axes[2].set_ylabel('R² Score')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(model_names, rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    best_idx = np.argmax(r2_values)
    bars3[best_idx].set_color('gold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_confusion_matrix_plot(predictions: np.ndarray,
                                targets: np.ndarray,
                                tolerance: float = 0.5,
                                save_path: str = "confusion_matrix.png",
                                show_plot: bool = False):
    """
    회귀 문제를 위한 허용 오차 기반 혼동 행렬 생성
    
    Args:
        predictions (np.ndarray): 예측값
        targets (np.ndarray): 실제값
        tolerance (float): 허용 오차 (Brix)
        save_path (str): 저장 경로
        show_plot (bool): 플롯 표시 여부
    """
    
    # 예측 정확성 분류
    errors = np.abs(predictions - targets)
    correct = errors <= tolerance
    
    # 당도 범위별 분류
    low_sweetness = targets < 10.0
    medium_sweetness = (targets >= 10.0) & (targets < 11.5)
    high_sweetness = targets >= 11.5
    
    # 혼동 행렬 데이터 생성
    categories = ['Low (<10)', 'Medium (10-11.5)', 'High (≥11.5)']
    confusion_data = np.zeros((3, 3))
    
    for i, target_mask in enumerate([low_sweetness, medium_sweetness, high_sweetness]):
        for j, pred_mask in enumerate([low_sweetness, medium_sweetness, high_sweetness]):
            # 실제 카테고리 i, 예측 카테고리 j인 샘플들 중 정확한 예측 비율
            mask = target_mask & pred_mask
            if np.sum(target_mask) > 0:
                confusion_data[i, j] = np.sum(mask & correct) / np.sum(target_mask) * 100
    
    # 혼동 행렬 플롯 생성
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(confusion_data, 
                annot=True, 
                fmt='.1f',
                xticklabels=categories,
                yticklabels=categories,
                cmap='Blues',
                cbar_kws={'label': 'Accuracy (%)'})
    
    plt.title(f'Prediction Accuracy by Sweetness Range\n(Tolerance: ±{tolerance} Brix)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('Actual Category', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


# 추가 유틸리티 함수들
def save_plots_as_pdf(plot_paths: List[str], pdf_path: str):
    """여러 플롯을 하나의 PDF로 결합"""
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(pdf_path) as pdf:
            for plot_path in plot_paths:
                if Path(plot_path).exists():
                    img = plt.imread(plot_path)
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(img)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
        
        print(f"📄 모든 플롯이 PDF로 저장됨: {pdf_path}")
        
    except ImportError:
        print("Warning: matplotlib.backends.backend_pdf를 사용할 수 없습니다.")
    except Exception as e:
        print(f"Warning: PDF 저장 중 오류 발생: {e}")


def create_interactive_plot(predictions: np.ndarray, 
                          targets: np.ndarray,
                          save_path: str = "interactive_plot.html"):
    """인터랙티브 플롯 생성 (plotly 사용)"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        
        # 산점도 생성
        fig = go.Figure()
        
        # 데이터 포인트
        fig.add_trace(go.Scatter(
            x=targets,
            y=predictions,
            mode='markers',
            marker=dict(
                size=8,
                color=np.abs(predictions - targets),
                colorscale='Viridis',
                colorbar=dict(title="Absolute Error"),
                opacity=0.7
            ),
            text=[f'Actual: {t:.2f}<br>Predicted: {p:.2f}<br>Error: {abs(p-t):.2f}' 
                  for t, p in zip(targets, predictions)],
            hovertemplate='%{text}<extra></extra>',
            name='Predictions'
        ))
        
        # 완벽한 예측 라인
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect Prediction'
        ))
        
        fig.update_layout(
            title='🍉 Interactive Watermelon Sweetness Prediction Results',
            xaxis_title='Actual Sweetness (Brix)',
            yaxis_title='Predicted Sweetness (Brix)',
            width=800,
            height=600
        )
        
        fig.write_html(save_path)
        print(f"📊 인터랙티브 플롯 저장됨: {save_path}")
        
    except ImportError:
        print("Warning: plotly가 설치되지 않아 인터랙티브 플롯을 생성할 수 없습니다.")
    except Exception as e:
        print(f"Warning: 인터랙티브 플롯 생성 중 오류 발생: {e}")


if __name__ == "__main__":
    # 테스트용 더미 데이터
    print("🧪 시각화 모듈 테스트")
    
    # 더미 데이터 생성
    np.random.seed(42)
    n_samples = 100
    targets = np.random.uniform(8.5, 12.5, n_samples)
    predictions = targets + np.random.normal(0, 0.3, n_samples)
    
    # 더미 평가 결과
    dummy_results = {
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'metrics': {
            'mae': 0.25,
            'rmse': 0.35,
            'r2_score': 0.85,
            'mape': 2.5,
            'sweetness_accuracy_05': 75.0,
            'sweetness_accuracy_10': 90.0
        }
    }
    
    # 플롯 생성 테스트
    plot_paths = create_evaluation_plots(dummy_results, "test_plots", show_plots=False)
    
    print(f"✅ {len(plot_paths)}개 테스트 플롯 생성 완료") 