#!/usr/bin/env python3
"""
실험 결과 비교 분석 스크립트
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Any
import numpy as np

def load_experiment_results(experiments_dir: Path) -> Dict[str, Dict]:
    """
    실험 결과들을 로드
    
    Args:
        experiments_dir: 실험 디렉토리 경로
        
    Returns:
        실험명별 결과 딕셔너리
    """
    results = {}
    
    # 실험 디렉토리들 스캔
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
            metrics_file = exp_dir / "metrics_summary.json"
            success_file = exp_dir / "SUCCESS"
            
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                    
                    # SUCCESS 파일에서 최종 결과 추출
                    final_results = {}
                    if success_file.exists():
                        with open(success_file, 'r') as f:
                            for line in f:
                                if "최고 Val MAE:" in line:
                                    final_results['best_val_mae'] = float(line.split(':')[1].strip())
                                elif "최종 테스트 MAE:" in line:
                                    final_results['final_test_mae'] = float(line.split(':')[1].strip())
                                elif "최종 테스트 R²:" in line:
                                    final_results['final_test_r2'] = float(line.split(':')[1].strip())
                    
                    # 실험명에서 설정 추출
                    exp_name = exp_dir.name
                    config_info = extract_config_from_name(exp_name)
                    
                    results[exp_name] = {
                        'metrics': data,
                        'final_results': final_results,
                        'config': config_info,
                        'status': 'completed' if success_file.exists() else 'in_progress'
                    }
                    
                except Exception as e:
                    print(f"⚠️ {exp_dir.name} 로드 실패: {e}")
    
    return results

def extract_config_from_name(exp_name: str) -> Dict[str, Any]:
    """
    실험명에서 설정 정보 추출
    
    Args:
        exp_name: 실험 디렉토리 명
        
    Returns:
        설정 정보 딕셔너리
    """
    config = {'experiment_type': 'unknown', 'batch_size': 8, 'loss_type': 'mse'}
    
    if 'batch16' in exp_name:
        config['experiment_type'] = 'batch_size'
        config['batch_size'] = 16
    elif 'batch32' in exp_name:
        config['experiment_type'] = 'batch_size'
        config['batch_size'] = 32
    elif 'huber' in exp_name:
        config['experiment_type'] = 'loss_function'
        config['loss_type'] = 'huber'
    elif 'regularized' in exp_name:
        config['experiment_type'] = 'regularization'
        config['regularization'] = 'enhanced'
    elif 'lr_reduced' in exp_name:
        config['experiment_type'] = 'learning_rate'
        config['learning_rate'] = 0.0001
    elif 'baseline' in exp_name:
        config['experiment_type'] = 'baseline'
        config['learning_rate'] = 0.001
    
    return config

def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    실험 결과 비교 테이블 생성
    
    Args:
        results: 실험 결과 딕셔너리
        
    Returns:
        비교 테이블 DataFrame
    """
    data = []
    
    for exp_name, exp_data in results.items():
        if exp_data['status'] == 'completed' and exp_data['final_results']:
            row = {
                'Experiment': exp_name,
                'Type': exp_data['config']['experiment_type'],
                'Batch Size': exp_data['config']['batch_size'],
                'Loss Type': exp_data['config']['loss_type'],
                'Best Val MAE': exp_data['final_results'].get('best_val_mae', 'N/A'),
                'Final Test MAE': exp_data['final_results'].get('final_test_mae', 'N/A'),
                'Final Test R²': exp_data['final_results'].get('final_test_r2', 'N/A'),
                'Status': exp_data['status']
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    return df.sort_values('Best Val MAE') if not df.empty else df

def plot_experiment_comparison(results: Dict[str, Dict], save_path: str = None):
    """
    실험 결과 시각화
    
    Args:
        results: 실험 결과 딕셔너리
        save_path: 저장 경로
    """
    # 완료된 실험들만 필터링
    completed_results = {k: v for k, v in results.items() 
                        if v['status'] == 'completed' and v['final_results']}
    
    if not completed_results:
        print("⚠️ 완료된 실험이 없습니다.")
        return
    
    # 데이터 준비
    exp_names = []
    val_maes = []
    test_maes = []
    r2_scores = []
    
    for exp_name, exp_data in completed_results.items():
        exp_names.append(exp_name.replace('_', '\n'))  # 줄바꿈으로 가독성 향상
        val_maes.append(exp_data['final_results']['best_val_mae'])
        test_maes.append(exp_data['final_results']['final_test_mae'])
        r2_scores.append(exp_data['final_results']['final_test_r2'])
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🍉 Watermelon Sweetness Prediction - Experiment Results Comparison', 
                 fontsize=16, fontweight='bold')
    
    # MAE 비교
    axes[0, 0].bar(range(len(exp_names)), val_maes, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Best Validation MAE', fontweight='bold')
    axes[0, 0].set_ylabel('MAE (Brix)')
    axes[0, 0].set_xticks(range(len(exp_names)))
    axes[0, 0].set_xticklabels(exp_names, rotation=45, ha='right')
    
    # Test MAE 비교
    axes[0, 1].bar(range(len(exp_names)), test_maes, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Final Test MAE', fontweight='bold')
    axes[0, 1].set_ylabel('MAE (Brix)')
    axes[0, 1].set_xticks(range(len(exp_names)))
    axes[0, 1].set_xticklabels(exp_names, rotation=45, ha='right')
    
    # R² Score 비교
    axes[1, 0].bar(range(len(exp_names)), r2_scores, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Final Test R² Score', fontweight='bold')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_xticks(range(len(exp_names)))
    axes[1, 0].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='R²=0')
    axes[1, 0].legend()
    
    # Val vs Test MAE 산점도
    axes[1, 1].scatter(val_maes, test_maes, s=100, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Validation MAE')
    axes[1, 1].set_ylabel('Test MAE')
    axes[1, 1].set_title('Validation vs Test MAE', fontweight='bold')
    
    # 대각선 추가 (이상적인 경우)
    min_mae = min(min(val_maes), min(test_maes))
    max_mae = max(max(val_maes), max(test_maes))
    axes[1, 1].plot([min_mae, max_mae], [min_mae, max_mae], 'r--', alpha=0.5, label='Perfect Fit')
    axes[1, 1].legend()
    
    # 실험명 라벨 추가
    for i, name in enumerate(exp_names):
        axes[1, 1].annotate(name.replace('\n', '_'), (val_maes[i], test_maes[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 비교 차트 저장됨: {save_path}")
    
    plt.show()

def generate_experiment_report(results: Dict[str, Dict], save_path: str = None) -> str:
    """
    실험 결과 리포트 생성
    
    Args:
        results: 실험 결과 딕셔너리
        save_path: 저장 경로
        
    Returns:
        리포트 텍스트
    """
    report = []
    report.append("# 🍉 Watermelon Sweetness Prediction - Experiment Results Report")
    report.append("=" * 70)
    report.append("")
    
    # 완료된 실험 요약
    completed = [k for k, v in results.items() if v['status'] == 'completed']
    in_progress = [k for k, v in results.items() if v['status'] == 'in_progress']
    
    report.append(f"## 📊 실험 현황")
    report.append(f"- ✅ 완료된 실험: {len(completed)}개")
    report.append(f"- 🔄 진행 중인 실험: {len(in_progress)}개")
    report.append("")
    
    # 최고 성능 실험
    if completed:
        best_exp = min(completed, key=lambda x: results[x]['final_results'].get('best_val_mae', float('inf')))
        best_mae = results[best_exp]['final_results']['best_val_mae']
        
        report.append(f"## 🏆 최고 성능 실험")
        report.append(f"- **실험명**: {best_exp}")
        report.append(f"- **Best Val MAE**: {best_mae:.4f}")
        report.append(f"- **실험 타입**: {results[best_exp]['config']['experiment_type']}")
        report.append("")
    
    # 실험별 상세 결과
    report.append("## 📋 실험별 상세 결과")
    
    for exp_name, exp_data in results.items():
        if exp_data['status'] == 'completed' and exp_data['final_results']:
            report.append(f"### {exp_name}")
            report.append(f"- **Status**: {exp_data['status']}")
            report.append(f"- **Type**: {exp_data['config']['experiment_type']}")
            report.append(f"- **Best Val MAE**: {exp_data['final_results']['best_val_mae']:.4f}")
            report.append(f"- **Final Test MAE**: {exp_data['final_results']['final_test_mae']:.4f}")
            report.append(f"- **Final Test R²**: {exp_data['final_results']['final_test_r2']:.4f}")
            report.append("")
    
    # 진행 중인 실험
    if in_progress:
        report.append("## 🔄 진행 중인 실험")
        for exp_name in in_progress:
            report.append(f"- {exp_name}: {results[exp_name]['config']['experiment_type']}")
        report.append("")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"📄 리포트 저장됨: {save_path}")
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description="실험 결과 비교 분석")
    parser.add_argument("--experiments-dir", type=str, default="experiments", 
                       help="실험 디렉토리 경로")
    parser.add_argument("--output-dir", type=str, default="experiments/comparison", 
                       help="출력 디렉토리")
    parser.add_argument("--show-plot", action="store_true", help="차트 표시")
    
    args = parser.parse_args()
    
    # 경로 설정
    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 실험 결과 로딩 중...")
    results = load_experiment_results(experiments_dir)
    
    if not results:
        print("❌ 실험 결과를 찾을 수 없습니다.")
        return
    
    print(f"📊 {len(results)}개 실험 발견")
    
    # 비교 테이블 생성
    df = create_comparison_table(results)
    if not df.empty:
        table_path = output_dir / "experiment_comparison.csv"
        df.to_csv(table_path, index=False)
        print(f"📊 비교 테이블 저장됨: {table_path}")
        print("\n" + df.to_string(index=False))
    
    # 시각화
    chart_path = output_dir / "experiment_comparison.png"
    plot_experiment_comparison(results, str(chart_path))
    
    # 리포트 생성
    report_path = output_dir / "experiment_report.md"
    report = generate_experiment_report(results, str(report_path))
    print("\n" + "="*50)
    print(report)

if __name__ == "__main__":
    main() 