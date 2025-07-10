#!/usr/bin/env python3
"""
수박 당도 예측 모델 실험 실행 스크립트
"""

import sys
import argparse
from pathlib import Path
import time
import yaml

# 상대 경로 설정
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from training.trainer import create_trainer_from_config

def main():
    parser = argparse.ArgumentParser(description="수박 당도 예측 모델 실험")
    parser.add_argument("--config", type=str, required=True, help="설정 파일 경로")
    parser.add_argument("--data", type=str, default="watermelon_sound_data", help="데이터 경로")
    parser.add_argument("--experiment-name", type=str, help="실험 이름")
    parser.add_argument("--epochs", type=int, help="에포크 수 (설정 파일 덮어쓰기)")
    
    args = parser.parse_args()
    
    # 경로 설정
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    
    print(f"🍉 수박 당도 예측 모델 실험 시작")
    print(f"   📋 설정 파일: {config_path}")
    print(f"   📁 데이터 경로: {data_path}")
    print("=" * 60)
    
    # 설정 파일 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 실험 이름 설정
    experiment_name = args.experiment_name or config.get('experiment', {}).get('name', 'experiment')
    
    # 에포크 수 덮어쓰기
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # 트레이너 생성
    trainer = create_trainer_from_config(str(config_path), str(data_path), experiment_name)
    
    # 훈련 설정
    training_config = config.get('training', {})
    
    # 훈련 시작
    start_time = time.time()
    
    try:
        results = trainer.train(
            num_epochs=training_config.get('epochs', 15),
            save_every=training_config.get('save_every', 5),
            validate_every=training_config.get('validate_every', 1),
            early_stopping=training_config.get('early_stopping', True),
            verbose=training_config.get('verbose', True)
        )
        
        # 결과 출력
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"🎉 실험 완료: {experiment_name}")
        print(f"   ⏱️ 총 시간: {total_time/3600:.2f}시간")
        print(f"   🏆 최고 Val MAE: {results['best_val_mae']:.4f}")
        print(f"   📊 최종 테스트 MAE: {results['final_test_metrics']['mae']:.4f}")
        print(f"   📊 최종 테스트 R²: {results['final_test_metrics']['r2_score']:.4f}")
        
        # 성공 상태 저장
        success_file = trainer.save_dir / "SUCCESS"
        with open(success_file, 'w') as f:
            f.write(f"실험 완료: {experiment_name}\n")
            f.write(f"최고 Val MAE: {results['best_val_mae']:.4f}\n")
            f.write(f"최종 테스트 MAE: {results['final_test_metrics']['mae']:.4f}\n")
            f.write(f"최종 테스트 R²: {results['final_test_metrics']['r2_score']:.4f}\n")
        
    except Exception as e:
        print(f"❌ 실험 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 