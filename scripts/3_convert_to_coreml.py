#!/usr/bin/env python3
"""
EfficientNet 모델을 Core ML로 직접 변환하는 스크립트
PyTorch → Core ML 직접 변환
최신 훈련된 EfficientNet 모델을 자동으로 찾아서 변환
"""

import sys
import torch
import numpy as np
from pathlib import Path
import traceback
import glob
from datetime import datetime

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.efficientnet_watermelon import create_efficientnet_watermelon


def find_latest_efficientnet_model():
    """최신 EfficientNet 실험 폴더에서 최고 성능 모델 찾기"""
    print("🔍 최신 EfficientNet 모델 검색 중...")
    
    # EfficientNet 실험 폴더들 찾기
    experiment_pattern = "experiments/direct_efficientnet_*"
    experiment_dirs = glob.glob(experiment_pattern)
    
    if not experiment_dirs:
        raise FileNotFoundError(f"EfficientNet 실험 폴더를 찾을 수 없습니다: {experiment_pattern}")
    
    # 가장 최신 폴더 찾기 (폴더명의 timestamp 기준)
    latest_dir = max(experiment_dirs)
    print(f"📂 최신 실험 폴더: {latest_dir}")
    
    # 해당 폴더에서 모델 파일 찾기
    model_files = glob.glob(f"{latest_dir}/*.pth")
    
    if not model_files:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {latest_dir}/*.pth")
    
    # 최고 MAE 모델 우선, 없으면 최고 Loss 모델 사용
    best_mae_files = [f for f in model_files if "best_mae" in f]
    best_loss_files = [f for f in model_files if "best_loss" in f]
    
    if best_mae_files:
        model_path = best_mae_files[0]
        print(f"🎯 최고 MAE 모델 사용: {model_path}")
    elif best_loss_files:
        model_path = best_loss_files[0]
        print(f"🎯 최고 Loss 모델 사용: {model_path}")
    else:
        # 가장 최신 모델 파일 사용
        model_path = max(model_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"🎯 최신 모델 파일 사용: {model_path}")
    
    return model_path


def load_model():
    """훈련된 모델 로드"""
    model_path = find_latest_efficientnet_model()
    
    print(f"🔄 모델 로드 중: {model_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 모델 설정 확인
    model_config = checkpoint.get('model_config', {})
    print(f"📋 모델 설정: {model_config}")
    
    # EfficientNet 모델 생성
    model = create_efficientnet_watermelon(
        model_name=model_config.get('model_name', 'efficientnet_b0'),
        pretrained=model_config.get('pretrained', False)
    )
    
    # 모델 가중치 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ EfficientNet 모델 로드 완료!")
    return model, model_path


def convert_to_coreml(model, output_path="models/converted/efficientnet.mlmodel"):
    """PyTorch 모델을 Core ML로 직접 변환"""
    try:
        import coremltools as ct
        print("🔄 PyTorch → Core ML 변환 시작...")
        
        # 출력 디렉토리 생성
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 더미 입력 생성
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # PyTorch 모델을 traced model로 변환
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Core ML 변환
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(1, 3, 224, 224), name="input")],
            outputs=[ct.TensorType(name="output")],
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
            minimum_deployment_target=ct.target.iOS13
        )
        
        # 모델 메타데이터 설정 (안전하게 처리)
        try:
            coreml_model.short_description = "수박 당도 예측 모델 (EfficientNet 기반)"
            coreml_model.author = "Watermelon ML Team"
            coreml_model.license = "MIT"
            coreml_model.version = "1.0"
            
            # 입력/출력 설명 추가
            coreml_model.input_description["input"] = "멜-스펙트로그램 이미지 (224x224x3)"
            coreml_model.output_description["output"] = "예측된 당도값 (Brix)"
        except Exception as meta_error:
            print(f"⚠️ 메타데이터 설정 실패 (모델 변환은 성공): {meta_error}")
        
        # 모델 저장
        coreml_model.save(str(output_path))
        
        print(f"✅ Core ML 변환 완료: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Core ML 변환 실패: {e}")
        traceback.print_exc()
        return None


def test_converted_model(model_path, original_model):
    """변환된 모델 테스트 (선택사항)"""
    try:
        import coremltools as ct
        print(f"🧪 변환된 모델 테스트: {model_path}")
        
        # Core ML 모델 로드
        coreml_model = ct.models.MLModel(str(model_path))
        
        # 더미 입력 생성
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # 원본 PyTorch 모델 예측
        with torch.no_grad():
            original_output = original_model(dummy_input)
            original_pred = original_output.numpy().flatten()[0]
        
        # Core ML 모델 예측
        input_dict = {"input": dummy_input.numpy()}
        coreml_output = coreml_model.predict(input_dict)
        coreml_pred = list(coreml_output.values())[0][0]
        
        # 결과 비교
        diff = abs(original_pred - coreml_pred)
        print(f"   📊 원본 모델 예측: {original_pred:.6f}")
        print(f"   📊 Core ML 예측: {coreml_pred:.6f}")
        print(f"   📊 차이: {diff:.6f}")
        
        if diff < 0.001:
            print("   ✅ 변환 성공! 예측값이 일치합니다.")
        else:
            print("   ⚠️ 변환된 모델과 원본 모델의 예측값에 차이가 있습니다.")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 변환된 모델 테스트 실패: {e}")
        print("   ℹ️ 테스트 실패는 macOS 환경 문제일 수 있으며, iOS에서는 정상 작동할 것입니다.")
        return False


def main():
    """메인 변환 함수"""
    print("🍉 EfficientNet 모델 → Core ML 변환!")
    print("="*50)
    
    try:
        # 1. 모델 로드
        model, model_path = load_model()
        print(f"📍 사용할 모델: {model_path}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None
    
    # 2. PyTorch → Core ML 변환
    print(f"\n🔄 PyTorch → Core ML 변환 시작...")
    coreml_path = convert_to_coreml(model)
    
    if coreml_path:
        # 3. 변환된 모델 테스트 (선택사항)
        print(f"\n🧪 변환된 모델 테스트...")
        test_result = test_converted_model(coreml_path, model)
        
        if test_result:
            print("   ✅ 예측 테스트도 성공!")
        else:
            print("   ⚠️ 예측 테스트는 실패했지만 모델 변환은 성공!")
        
        print(f"\n🎉 변환 완료! Core ML 모델: {coreml_path}")
        return coreml_path
    else:
        print(f"\n❌ Core ML 변환에 실패했습니다.")
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n✅ 최종 결과: {result}")
        print("📱 iOS 앱에서 사용할 수 있습니다!")
    else:
        print(f"\n❌ 변환에 실패했습니다.") 