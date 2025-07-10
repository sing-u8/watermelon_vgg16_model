#!/usr/bin/env python3
"""
Watermelon Sweetness Prediction - Model Training Script
수박 당도 예측 모델 훈련 메인 스크립트
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

# 커스텀 모듈 import
from src.training.trainer import create_trainer_from_config
from src.evaluation.evaluator import WatermelonEvaluator


def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="🍉 Watermelon Sweetness Prediction Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python scripts/train_model.py --config configs/training.yaml
  python scripts/train_model.py --config configs/training.yaml --experiment baseline_v1
  python scripts/train_model.py --config configs/training.yaml --epochs 50 --batch-size 16
        """
    )
    
    # 필수 인자
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="훈련 설정 파일 경로 (기본값: configs/training.yaml)"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="watermelon_sound_data",
        help="수박 오디오 데이터 경로 (기본값: watermelon_sound_data)"
    )
    
    # 선택적 인자 (설정 파일 오버라이드)
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="실험 이름 (기본값: 설정 파일의 experiment.name)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="훈련 에포크 수 (기본값: 설정 파일 값)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="배치 크기 (기본값: 설정 파일 값)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="학습률 (기본값: 설정 파일 값)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="훈련 디바이스 (기본값: auto)"
    )
    
    # 기능 플래그
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="체크포인트 경로 (훈련 재개용)"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="훈련 없이 평가만 수행"
    )
    
    parser.add_argument(
        "--save-dir",
        type=str,
        default="experiments",
        help="실험 결과 저장 디렉토리 (기본값: experiments)"
    )
    
    return parser.parse_args()


def setup_environment(config: dict, args):
    """환경 설정 및 시드 고정"""
    # 경고 필터링
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
    warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
    
    # 재현성 설정
    repro_config = config.get('reproducibility', {})
    if repro_config.get('deterministic', True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 시드 설정
    seed = config.get('data', {}).get('random_seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    # 디바이스 설정
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ 디바이스: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    return device


def override_config(config: dict, args):
    """명령행 인자로 설정 오버라이드"""
    # 에포크 수 오버라이드
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = args.epochs
    
    # 배치 크기 오버라이드
    if args.batch_size is not None:
        config.setdefault('data', {})['batch_size'] = args.batch_size
    
    # 학습률 오버라이드
    if args.learning_rate is not None:
        config.setdefault('optimizer', {})['lr'] = args.learning_rate
    
    # 실험 이름 오버라이드
    if args.experiment is not None:
        config.setdefault('experiment', {})['name'] = args.experiment
    
    return config


def load_config(config_path: str):
    """설정 파일 로드"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """메인 함수"""
    print("🍉" + "="*60)
    print("       WATERMELON SWEETNESS PREDICTION - TRAINING")
    print("🍉" + "="*60)
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 명령행 인자 파싱
    args = parse_arguments()
    
    # 데이터 경로 확인
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ 데이터 경로를 찾을 수 없습니다: {data_path}")
        return
    
    print(f"📁 데이터 경로: {data_path.absolute()}")
    
    try:
        # 설정 파일 로드
        print(f"⚙️ 설정 파일 로드: {args.config}")
        config = load_config(args.config)
        
        # 명령행 인자로 설정 오버라이드
        config = override_config(config, args)
        
        # 환경 설정
        device = setup_environment(config, args)
        
        # 실험 이름 설정
        experiment_name = config.get('experiment', {}).get('name', 'watermelon_training')
        experiment_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🧪 실험 이름: {experiment_name}")
        
        # 저장 디렉토리 생성
        save_dir = Path(args.save_dir) / experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 설정 파일 백업
        config_backup_path = save_dir / "config.yaml"
        with open(config_backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"📄 설정 파일 백업: {config_backup_path}")
        
        # 트레이너 생성
        print(f"\n🚂 트레이너 생성 중...")
        trainer = create_trainer_from_config(
            config_path=args.config,
            data_path=str(data_path),
            experiment_name=experiment_name
        )
        
        # 평가만 수행하는 경우
        if args.eval_only:
            print(f"\n🔍 평가 모드 - 훈련 없이 모델 평가만 수행")
            if args.resume is None:
                print("❌ 평가 모드에서는 --resume 옵션으로 모델 경로를 지정해야 합니다.")
                return
            
            # 모델 로드 및 평가
            trainer.model.load_state_dict(torch.load(args.resume, map_location=device))
            test_metrics = trainer.test_model()
            
            print(f"\n📊 최종 테스트 결과:")
            for metric, value in test_metrics.items():
                print(f"   {metric}: {value:.4f}")
            
            return
        
        # 훈련 재개하는 경우
        if args.resume is not None:
            print(f"🔄 훈련 재개: {args.resume}")
            # 모델 가중치 로드
            trainer.model.load_state_dict(torch.load(args.resume, map_location=device))
        
        # 훈련 파라미터 출력
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        optimizer_config = config.get('optimizer', {})
        
        print(f"\n📊 훈련 설정:")
        print(f"   에포크: {training_config.get('epochs', 100)}")
        print(f"   배치 크기: {data_config.get('batch_size', 32)}")
        print(f"   학습률: {optimizer_config.get('lr', 0.001)}")
        print(f"   옵티마이저: {optimizer_config.get('type', 'adam').upper()}")
        print(f"   조기 종료: {training_config.get('early_stopping', True)}")
        
        # 훈련 시작
        print(f"\n🚀 훈련 시작!")
        print("="*60)
        
        results = trainer.train(
            num_epochs=training_config.get('epochs', 100),
            save_every=training_config.get('save_every', 5),
            validate_every=training_config.get('validate_every', 1),
            early_stopping=training_config.get('early_stopping', True),
            verbose=training_config.get('verbose', True)
        )
        
        # 훈련 완료 결과 출력
        print("="*60)
        print(f"✅ 훈련 완료!")
        print(f"   🏆 최고 Val MAE: {results['best_val_mae']:.4f}")
        print(f"   📊 총 에포크: {results['total_epochs']}")
        print(f"   ⏱️ 훈련 시간: {results['training_time']/3600:.2f}시간")
        
        # 최종 테스트 결과
        final_test = results.get('final_test_metrics', {})
        if final_test:
            print(f"\n📈 최종 테스트 결과:")
            for metric, value in final_test.items():
                print(f"   {metric}: {value:.4f}")
        
        # 모델 평가 수행
        print(f"\n🔍 상세 모델 평가 수행 중...")
        evaluator = WatermelonEvaluator(
            model=trainer.model,
            device=device,
            save_dir=str(save_dir)
        )
        
        evaluation_results = evaluator.evaluate_model(trainer.data_loader, "test")
        
        print(f"📊 평가 결과 저장 완료: {save_dir}")
        print(f"📁 훈련 결과:")
        print(f"   - 모델 체크포인트: {trainer.best_model_path}")
        print(f"   - 훈련 곡선: {save_dir}/training_curves.png")
        print(f"   - 평가 리포트: {save_dir}/evaluation_report.md")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 훈련이 중단되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n⏰ 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🍉" + "="*60)


if __name__ == "__main__":
    main() 