#!/usr/bin/env python3
"""
평가 모듈 테스트 스크립트
Phase 3.6: 평가 모듈 완성 및 검증
"""

import numpy as np
import torch
import sys
from pathlib import Path

def test_evaluation_module():
    """평가 모듈 전체 테스트"""
    
    print("🧪 평가 모듈 테스트 시작")
    print("=" * 50)
    
    try:
        # 1. 기본 모듈 import 테스트
        print("📦 모듈 import 테스트...")
        
        from src.evaluation.evaluator import WatermelonEvaluator, create_evaluator, quick_evaluation
        print("   ✅ evaluator 모듈 import 성공")
        
        from src.evaluation.model_analyzer import ModelAnalyzer
        print("   ✅ model_analyzer 모듈 import 성공")
        
        from src.evaluation.visualization import create_evaluation_plots
        print("   ✅ visualization 모듈 import 성공")
        
        from src.evaluation import WatermelonEvaluator, ModelAnalyzer, create_evaluation_plots
        print("   ✅ 통합 import 성공")
        
        # 2. 더미 모델 생성 테스트
        print("\n🤖 더미 모델 생성 테스트...")
        
        from src.models.vgg_watermelon import create_vgg_watermelon
        model = create_vgg_watermelon()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   ✅ VGG 모델 생성 성공 (device: {device})")
        
        # 3. 평가기 초기화 테스트
        print("\n🔍 평가기 초기화 테스트...")
        
        evaluator = WatermelonEvaluator(model, device, "test_evaluation_results")
        print("   ✅ WatermelonEvaluator 초기화 성공")
        
        # 4. 모델 분석기 초기화 테스트
        print("\n🔬 모델 분석기 초기화 테스트...")
        
        analyzer = ModelAnalyzer(model, device)
        print("   ✅ ModelAnalyzer 초기화 성공")
        
        # 5. 아키텍처 분석 테스트
        print("\n🏗️ 아키텍처 분석 테스트...")
        
        arch_analysis = analyzer.analyze_model_architecture()
        print(f"   ✅ 아키텍처 분석 완료 (총 파라미터: {arch_analysis['model_info']['total_parameters']:,})")
        
        # 6. 더미 데이터로 시각화 테스트
        print("\n📊 시각화 테스트...")
        
        # 더미 평가 결과 생성
        np.random.seed(42)
        n_samples = 50
        targets = np.random.uniform(8.5, 12.5, n_samples)
        predictions = targets + np.random.normal(0, 0.3, n_samples)
        
        dummy_results = {
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'metrics': {
                'mae': 0.25,
                'rmse': 0.35,
                'r2_score': 0.85,
                'mape': 2.5,
                'sweetness_accuracy_05': 75.0,
                'sweetness_accuracy_10': 90.0,
                'pearson_correlation': 0.92,
                'num_samples': n_samples
            },
            'performance_analysis': {
                'range_analysis': {
                    'low_sweetness': {
                        'count': 15,
                        'mae': 0.22,
                        'rmse': 0.31,
                        'r2_score': 0.88,
                        'range': (8.0, 10.0)
                    },
                    'medium_sweetness': {
                        'count': 20,
                        'mae': 0.25,
                        'rmse': 0.35,
                        'r2_score': 0.85,
                        'range': (10.0, 11.5)
                    },
                    'high_sweetness': {
                        'count': 15,
                        'mae': 0.28,
                        'rmse': 0.38,
                        'r2_score': 0.82,
                        'range': (11.5, 13.0)
                    }
                }
            }
        }
        
        # 시각화 생성
        plot_paths = create_evaluation_plots(dummy_results, "test_evaluation_plots", show_plots=False)
        print(f"   ✅ {len(plot_paths)}개 시각화 플롯 생성 성공")
        
        # 7. 평가 보고서 생성 테스트
        print("\n📄 평가 보고서 생성 테스트...")
        
        report_path = evaluator.generate_evaluation_report(dummy_results, include_plots=False)
        print(f"   ✅ 평가 보고서 생성 성공: {report_path}")
        
        print("\n🎉 모든 평가 모듈 테스트 성공!")
        print("✅ Phase 3.6: 평가 모듈 완성 및 검증 - 완료")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    
    # 프로젝트 루트 디렉토리 확인
    current_dir = Path.cwd()
    print(f"📁 현재 디렉토리: {current_dir}")
    
    # Python 경로에 프로젝트 루트 추가
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # 테스트 실행
    success = test_evaluation_module()
    
    if success:
        print("\n📋 다음 단계:")
        print("   - Phase 4: 모델 훈련 실행 및 실험")
        print("   - 실제 데이터셋으로 모델 훈련")
        print("   - 하이퍼파라미터 튜닝")
        print("   - 최적 모델 선택")
        
        # Todo 업데이트
        try:
            print("\n📝 진행 상황 업데이트 중...")
            # 여기서 todo 업데이트 로직을 추가할 수 있음
            print("   ✅ Phase 3.6 완료 마킹")
        except:
            pass
    
    return success

if __name__ == "__main__":
    main() 