#!/usr/bin/env python3
"""
모델 변환 스크립트: PyTorch → ONNX → Core ML
iOS 배포를 위한 모델 변환 및 최적화
"""

import torch
import torch.onnx
import onnx
try:
    import onnxsim
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    print("⚠️ onnxsim not available, skipping model simplification")
import coremltools as ct
import numpy as np
from pathlib import Path
import argparse
import json
import time
from typing import Tuple, Dict, Any
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vgg_watermelon import VGGWatermelon

class ModelConverter:
    """모델 변환 클래스"""
    
    def __init__(self, model_path: str, output_dir: str = "models/converted"):
        """
        초기화
        
        Args:
            model_path: PyTorch 모델 파일 경로
            output_dir: 변환된 모델 저장 디렉토리
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        
        # 변환 결과 저장용
        self.conversion_results = {}
        
    def _load_model(self) -> torch.nn.Module:
        """PyTorch 모델 로드"""
        try:
            print(f"🔄 PyTorch 모델 로딩: {self.model_path}")
            
            # 체크포인트 로드 (PyTorch 2.6+ 호환성)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # 체크포인트에서 모델 설정 추출
            model_config = checkpoint.get('model_config', {})
            model_name = model_config.get('model_name', 'VGGWatermelon')
            backbone = model_config.get('backbone', 'VGG-16')
            fc_hidden_size = model_config.get('fc_hidden_size', 512)
            dropout_rate = model_config.get('dropout_rate', 0.5)
            
            print(f"📊 모델 정보: {model_name} (백본: {backbone})")
            print(f"📊 모델 설정: FC Hidden Size={fc_hidden_size}, Dropout={dropout_rate}")
            
            # 모델 타입에 따라 적절한 모델 생성
            if 'EfficientNet' in model_name:
                print("🔧 EfficientNet 모델 로딩...")
                from src.models.efficientnet_watermelon import create_efficientnet_watermelon
                model = create_efficientnet_watermelon(
                    pretrained=False,
                    dropout_rate=dropout_rate,
                    num_fc_layers=model_config.get('fc_layers', 2),
                    fc_hidden_size=fc_hidden_size
                )
            elif 'MelSpecCNN' in model_name:
                print("🔧 MelSpecCNN 모델 로딩...")
                from src.models.melspec_cnn_watermelon import create_melspec_cnn_watermelon
                model = create_melspec_cnn_watermelon(
                    input_channels=model_config.get('input_channels', 3),
                    base_channels=model_config.get('base_channels', 32),
                    dropout_rate=dropout_rate
                )
            else:
                print("🔧 VGG 모델 로딩...")
                # 기존 VGG 모델 로딩
                model = VGGWatermelon(
                    input_channels=3,
                    pretrained=False,  # 이미 훈련된 모델이므로 False
                    fc_hidden_size=fc_hidden_size,
                    dropout_rate=dropout_rate
                )
            
            # 가중치 로드
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 모델 상태 로드 완료 (에포크: {checkpoint.get('epoch', 'N/A')})")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ 모델 상태 로드 완료")
            
            model.eval()
            model.to(self.device)
            
            # 모델 정보 출력
            total_params = sum(p.numel() for p in model.parameters())
            print(f"📊 모델 파라미터 수: {total_params:,}")
            
            return model
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def create_dummy_input(self, batch_size: int = 1) -> torch.Tensor:
        """
        더미 입력 생성 (멜-스펙트로그램 형태)
        
        Args:
            batch_size: 배치 크기
            
        Returns:
            더미 입력 텐서 (B, C, H, W)
        """
        # VGG-16 입력 크기: 224x224x3
        input_shape = (batch_size, 3, 224, 224)
        dummy_input = torch.randn(input_shape, device=self.device)
        
        print(f"📋 더미 입력 크기: {input_shape}")
        return dummy_input
    
    def convert_to_onnx(self, 
                       output_name: str = None,
                       opset_version: int = 11,
                       simplify: bool = True,
                       batch_size: int = 1) -> str:
        """
        PyTorch → ONNX 변환
        
        Args:
            output_name: 출력 파일명
            opset_version: ONNX opset 버전
            simplify: ONNX 모델 최적화 여부
            batch_size: 배치 크기
            
        Returns:
            변환된 ONNX 파일 경로
        """
        try:
            if output_name is None:
                output_name = f"{self.model_path.stem}_batch{batch_size}.onnx"
            
            onnx_path = self.output_dir / output_name
            
            print(f"🔄 ONNX 변환 시작...")
            print(f"   - 출력 경로: {onnx_path}")
            print(f"   - Opset 버전: {opset_version}")
            print(f"   - 배치 크기: {batch_size}")
            
            # 더미 입력 생성
            dummy_input = self.create_dummy_input(batch_size)
            
            # 동적 배치 크기 설정
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            
            # ONNX 변환
            start_time = time.time()
            
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            conversion_time = time.time() - start_time
            
            # ONNX 모델 검증
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 모델 최적화 (ONNX Simplifier)
            if simplify and ONNXSIM_AVAILABLE:
                print(f"🔧 ONNX 모델 최적화 중...")
                try:
                    simplified_model, check = onnxsim.simplify(onnx_model)
                    if check:
                        onnx.save(simplified_model, str(onnx_path))
                        print(f"✅ ONNX 모델 최적화 완료")
                    else:
                        print(f"⚠️ ONNX 모델 최적화 실패, 원본 사용")
                except Exception as e:
                    print(f"⚠️ ONNX 최적화 건너뜀: {e}")
            elif simplify and not ONNXSIM_AVAILABLE:
                print(f"⚠️ ONNX Simplifier 미설치, 최적화 건너뜀")
            
            # 파일 크기 확인
            file_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
            
            print(f"✅ ONNX 변환 완료!")
            print(f"   - 변환 시간: {conversion_time:.2f}초")
            print(f"   - 파일 크기: {file_size:.2f}MB")
            
            # 결과 저장
            self.conversion_results['onnx'] = {
                'path': str(onnx_path),
                'size_mb': file_size,
                'conversion_time': conversion_time,
                'opset_version': opset_version,
                'batch_size': batch_size
            }
            
            return str(onnx_path)
            
        except Exception as e:
            print(f"❌ ONNX 변환 실패: {e}")
            raise
    
    def convert_to_coreml(self, 
                         onnx_path: str = None,
                         output_name: str = None,
                         compute_units: str = "ALL",
                         quantize: bool = True) -> str:
        """
        ONNX → Core ML 변환
        
        Args:
            onnx_path: ONNX 모델 경로
            output_name: 출력 파일명
            compute_units: 연산 유닛 (ALL, CPU_ONLY, CPU_AND_GPU)
            quantize: 양자화 적용 여부
            
        Returns:
            변환된 Core ML 파일 경로
        """
        try:
            if onnx_path is None:
                # 가장 최근 변환된 ONNX 파일 사용
                onnx_files = list(self.output_dir.glob("*.onnx"))
                if not onnx_files:
                    raise ValueError("ONNX 파일을 찾을 수 없습니다. 먼저 ONNX 변환을 실행하세요.")
                onnx_path = max(onnx_files, key=lambda x: x.stat().st_mtime)
            
            if output_name is None:
                output_name = f"{Path(onnx_path).stem}.mlmodel"
            
            coreml_path = self.output_dir / output_name
            
            print(f"🔄 Core ML 변환 시작...")
            print(f"   - ONNX 경로: {onnx_path}")
            print(f"   - 출력 경로: {coreml_path}")
            print(f"   - 연산 유닛: {compute_units}")
            print(f"   - 양자화: {quantize}")
            
            start_time = time.time()
            
            # Core ML 변환
            # 입력 형태 정의
            input_shape = ct.Shape(shape=(1, 3, 224, 224))  # (batch, channels, height, width)
            
            # 변환 옵션 설정
            convert_options = {
                'inputs': [ct.TensorType(name='input', shape=input_shape)],
                'outputs': [ct.TensorType(name='output')],
                'compute_units': getattr(ct.ComputeUnit, compute_units),
            }
            
            # 양자화 설정
            if quantize:
                # 16-bit 양자화 (iOS에서 권장)
                convert_options['compute_precision'] = ct.precision.FLOAT16
            
            # ONNX 모델 로드 및 변환
            coreml_model = ct.convert(
                onnx_path,
                source="onnx",
                **convert_options
            )
            
            # 모델 메타데이터 설정
            coreml_model.short_description = "🍉 Watermelon Sweetness Prediction Model"
            coreml_model.input_description['input'] = "Mel-spectrogram of watermelon tapping sound (224x224x3)"
            coreml_model.output_description['output'] = "Predicted sweetness value in Brix"
            
            # 버전 정보
            coreml_model.version = "1.0"
            coreml_model.author = "Watermelon ML Team"
            
            # 모델 저장
            coreml_model.save(str(coreml_path))
            
            conversion_time = time.time() - start_time
            
            # 파일 크기 확인
            file_size = coreml_path.stat().st_size / (1024 * 1024)  # MB
            
            print(f"✅ Core ML 변환 완료!")
            print(f"   - 변환 시간: {conversion_time:.2f}초")
            print(f"   - 파일 크기: {file_size:.2f}MB")
            
            # 모델 검증
            self._validate_coreml_model(str(coreml_path))
            
            # 결과 저장
            self.conversion_results['coreml'] = {
                'path': str(coreml_path),
                'size_mb': file_size,
                'conversion_time': conversion_time,
                'compute_units': compute_units,
                'quantized': quantize
            }
            
            return str(coreml_path)
            
        except Exception as e:
            print(f"❌ Core ML 변환 실패: {e}")
            raise
    
    def _validate_coreml_model(self, coreml_path: str):
        """Core ML 모델 검증"""
        try:
            print(f"🔍 Core ML 모델 검증 중...")
            
            # Core ML 모델 로드
            model = ct.models.MLModel(coreml_path)
            
            # 모델 정보 출력
            spec = model.get_spec()
            print(f"   - 모델 타입: {spec.WhichOneof('Type')}")
            print(f"   - Core ML 버전: {spec.specificationVersion}")
            
            # 입력/출력 정보
            for input_desc in spec.description.input:
                print(f"   - 입력: {input_desc.name} {input_desc.type}")
            
            for output_desc in spec.description.output:
                print(f"   - 출력: {output_desc.name} {output_desc.type}")
            
            print(f"✅ Core ML 모델 검증 완료")
            
        except Exception as e:
            print(f"⚠️ Core ML 모델 검증 실패: {e}")
    
    def test_inference(self, onnx_path: str = None, coreml_path: str = None) -> Dict[str, Any]:
        """
        변환된 모델들의 추론 성능 테스트
        
        Args:
            onnx_path: ONNX 모델 경로
            coreml_path: Core ML 모델 경로
            
        Returns:
            추론 성능 결과
        """
        results = {}
        
        # 테스트 입력 준비
        test_input = self.create_dummy_input(batch_size=1)
        test_input_np = test_input.cpu().numpy()
        
        print(f"🧪 추론 성능 테스트 중...")
        
        # 1. PyTorch 모델 테스트
        try:
            with torch.no_grad():
                start_time = time.time()
                pytorch_output = self.model(test_input)
                pytorch_time = time.time() - start_time
                
                results['pytorch'] = {
                    'inference_time': pytorch_time,
                    'output_shape': list(pytorch_output.shape),
                    'output_sample': float(pytorch_output[0].item())
                }
                
                print(f"   ✅ PyTorch: {pytorch_time*1000:.2f}ms")
                
        except Exception as e:
            print(f"   ❌ PyTorch 테스트 실패: {e}")
        
        # 2. ONNX 모델 테스트
        if onnx_path and Path(onnx_path).exists():
            try:
                import onnxruntime as ort
                
                session = ort.InferenceSession(onnx_path)
                
                start_time = time.time()
                onnx_output = session.run(['output'], {'input': test_input_np})
                onnx_time = time.time() - start_time
                
                results['onnx'] = {
                    'inference_time': onnx_time,
                    'output_shape': list(onnx_output[0].shape),
                    'output_sample': float(onnx_output[0][0])
                }
                
                print(f"   ✅ ONNX: {onnx_time*1000:.2f}ms")
                
            except Exception as e:
                print(f"   ❌ ONNX 테스트 실패: {e}")
        
        # 3. Core ML 모델 테스트
        if coreml_path and Path(coreml_path).exists():
            try:
                import coremltools as ct
                
                model = ct.models.MLModel(coreml_path)
                
                # 입력 데이터 준비 (Core ML 형식)
                input_dict = {'input': test_input_np}
                
                start_time = time.time()
                coreml_output = model.predict(input_dict)
                coreml_time = time.time() - start_time
                
                output_value = list(coreml_output.values())[0]
                
                results['coreml'] = {
                    'inference_time': coreml_time,
                    'output_shape': list(output_value.shape) if hasattr(output_value, 'shape') else [1],
                    'output_sample': float(output_value[0]) if hasattr(output_value, '__getitem__') else float(output_value)
                }
                
                print(f"   ✅ Core ML: {coreml_time*1000:.2f}ms")
                
            except Exception as e:
                print(f"   ❌ Core ML 테스트 실패: {e}")
        
        return results
    
    def save_conversion_report(self, inference_results: Dict = None):
        """변환 결과 리포트 저장"""
        report = {
            'model_info': {
                'source_model': str(self.model_path),
                'conversion_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device)
            },
            'conversion_results': self.conversion_results,
            'inference_results': inference_results or {}
        }
        
        report_path = self.output_dir / 'conversion_report.json'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 변환 리포트 저장: {report_path}")
        
        # 요약 출력
        print(f"\n{'='*50}")
        print(f"🎯 모델 변환 완료 요약")
        print(f"{'='*50}")
        
        for format_name, result in self.conversion_results.items():
            print(f"📱 {format_name.upper()}:")
            print(f"   - 파일: {result['path']}")
            print(f"   - 크기: {result['size_mb']:.2f}MB")
            print(f"   - 변환 시간: {result['conversion_time']:.2f}초")
        
        if inference_results:
            print(f"\n⚡ 추론 성능 비교:")
            for format_name, result in inference_results.items():
                time_ms = result['inference_time'] * 1000
                print(f"   - {format_name.upper()}: {time_ms:.2f}ms")

def main():
    parser = argparse.ArgumentParser(description="PyTorch 모델을 ONNX와 Core ML로 변환")
    parser.add_argument("model_path", help="PyTorch 모델 파일 경로 (.pth)")
    parser.add_argument("--output-dir", default="models/converted", help="출력 디렉토리")
    parser.add_argument("--batch-size", type=int, default=1, help="배치 크기")
    parser.add_argument("--opset-version", type=int, default=11, help="ONNX opset 버전")
    parser.add_argument("--compute-units", default="ALL", 
                       choices=["ALL", "CPU_ONLY", "CPU_AND_GPU"],
                       help="Core ML 연산 유닛")
    parser.add_argument("--no-quantize", action="store_true", help="Core ML 양자화 비활성화")
    parser.add_argument("--no-simplify", action="store_true", help="ONNX 최적화 비활성화")
    parser.add_argument("--test-inference", action="store_true", help="추론 성능 테스트 실행")
    parser.add_argument("--onnx-only", action="store_true", help="ONNX 변환만 실행")
    parser.add_argument("--coreml-only", action="store_true", help="Core ML 변환만 실행")
    
    args = parser.parse_args()
    
    # 변환기 초기화
    converter = ModelConverter(args.model_path, args.output_dir)
    
    onnx_path = None
    coreml_path = None
    
    try:
        # ONNX 변환
        if not args.coreml_only:
            onnx_path = converter.convert_to_onnx(
                batch_size=args.batch_size,
                opset_version=args.opset_version,
                simplify=not args.no_simplify
            )
        
        # Core ML 변환
        if not args.onnx_only:
            coreml_path = converter.convert_to_coreml(
                onnx_path=onnx_path,
                compute_units=args.compute_units,
                quantize=not args.no_quantize
            )
        
        # 추론 성능 테스트
        inference_results = None
        if args.test_inference:
            inference_results = converter.test_inference(onnx_path, coreml_path)
        
        # 리포트 저장
        converter.save_conversion_report(inference_results)
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 