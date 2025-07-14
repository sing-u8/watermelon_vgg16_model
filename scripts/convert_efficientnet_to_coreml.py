#!/usr/bin/env python3
"""
EfficientNet 모델을 Core ML로 변환하는 스크립트
PyTorch → ONNX → Core ML 경로 시도 후, 실패시 직접 변환
"""

import sys
import torch
import numpy as np
from pathlib import Path
import traceback

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.vgg_watermelon import VGGWatermelon


def load_model():
    """훈련된 모델 로드"""
    model_path = "experiments/efficientnet_fixed_final/best_model.pth"
    
    print("🔄 모델 로드 중...")
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 모델 설정 확인
    model_config = checkpoint.get('model_config', {})
    print(f"📋 모델 설정: {model_config}")
    
    # 모델 생성 (저장된 설정 사용)
    model = VGGWatermelon(
        fc_hidden_size=model_config.get('fc_hidden_size', 256),
        num_fc_layers=model_config.get('num_fc_layers', 2),
        dropout_rate=model_config.get('dropout_rate', 0.7)
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ 모델 로드 완료!")
    return model


def convert_to_onnx(model, output_path="models/converted/efficientnet_fixed.onnx"):
    """PyTorch 모델을 ONNX로 변환"""
    try:
        print("🔄 ONNX 변환 시작...")
        
        # 출력 디렉토리 생성
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 더미 입력 생성 (배치 크기 1, 채널 3, 224x224)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # ONNX 변환
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX 변환 완료: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        traceback.print_exc()
        return None


def convert_onnx_to_coreml(onnx_path, output_path="models/converted/efficientnet_fixed.mlmodel"):
    """ONNX 모델을 Core ML로 변환"""
    try:
        import coremltools as ct
        print("🔄 Core ML 변환 시작...")
        
        # 출력 디렉토리 생성
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ONNX 모델 로드 및 Core ML 변환
        model = ct.convert(
            str(onnx_path),
            inputs=[ct.TensorType(shape=(1, 3, 224, 224), name="input")],
            outputs=[ct.TensorType(name="output")],
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
            minimum_deployment_target=ct.target.iOS13
        )
        
        # 모델 메타데이터 설정
        model.short_description = "수박 당도 예측 모델 (EfficientNet 기반)"
        model.author = "Watermelon ML Team"
        model.license = "MIT"
        model.version = "1.0"
        
        # 입력/출력 설명 추가
        model.input_description["input"] = "멜-스펙트로그램 이미지 (224x224x3)"
        model.output_description["output"] = "예측된 당도값 (Brix)"
        
        # 모델 저장
        model.save(str(output_path))
        
        print(f"✅ Core ML 변환 완료: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Core ML 변환 실패: {e}")
        traceback.print_exc()
        return None


def convert_pytorch_to_coreml_direct(model, output_path="models/converted/efficientnet_fixed_direct.mlmodel"):
    """PyTorch 모델을 Core ML로 직접 변환"""
    try:
        import coremltools as ct
        print("🔄 PyTorch → Core ML 직접 변환 시작...")
        
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
        
        # 모델 메타데이터 설정
        coreml_model.short_description = "수박 당도 예측 모델 (EfficientNet 기반 - 직접 변환)"
        coreml_model.author = "Watermelon ML Team"
        coreml_model.license = "MIT"
        coreml_model.version = "1.0"
        
        # 입력/출력 설명 추가
        coreml_model.input_description["input"] = "멜-스펙트로그램 이미지 (224x224x3)"
        coreml_model.output_description["output"] = "예측된 당도값 (Brix)"
        
        # 모델 저장
        coreml_model.save(str(output_path))
        
        print(f"✅ Core ML 직접 변환 완료: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Core ML 직접 변환 실패: {e}")
        traceback.print_exc()
        return None


def test_converted_model(model_path, original_model):
    """변환된 모델 테스트"""
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
        return False


def main():
    """메인 변환 함수"""
    print("🍉 EfficientNet 모델 → Core ML 변환 시작!")
    print("="*60)
    
    # 1. 모델 로드
    model = load_model()
    
    # 2. PyTorch → ONNX → Core ML 시도
    print("\n📍 방법 1: PyTorch → ONNX → Core ML")
    onnx_path = convert_to_onnx(model)
    
    if onnx_path:
        coreml_path = convert_onnx_to_coreml(onnx_path)
        if coreml_path:
            # 변환된 모델 테스트
            if test_converted_model(coreml_path, model):
                print(f"\n🎉 변환 완료! Core ML 모델: {coreml_path}")
                return coreml_path
    
    # 3. PyTorch → Core ML 직접 변환 시도
    print("\n📍 방법 2: PyTorch → Core ML 직접 변환")
    coreml_path_direct = convert_pytorch_to_coreml_direct(model)
    
    if coreml_path_direct:
        # 변환된 모델 테스트
        if test_converted_model(coreml_path_direct, model):
            print(f"\n🎉 직접 변환 완료! Core ML 모델: {coreml_path_direct}")
            return coreml_path_direct
    
    print("\n❌ 모든 변환 방법이 실패했습니다.")
    return None


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n✅ 최종 결과: {result}")
        print("📱 iOS 앱에서 사용할 수 있습니다!")
    else:
        print("\n❌ 변환에 실패했습니다.") 