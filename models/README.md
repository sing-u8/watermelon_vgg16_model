# 💾 Saved Models

훈련 완료된 모델들을 다양한 형식으로 저장합니다.

## 모델 형식

### PyTorch Models (.pth)
- `watermelon_model.pth` - 훈련된 PyTorch 모델
- `watermelon_model_state_dict.pth` - state dictionary만
- `watermelon_model_full.pth` - 전체 모델 객체

### ONNX Models (.onnx)
- `watermelon_model.onnx` - ONNX 변환 모델
- `watermelon_model_optimized.onnx` - 최적화된 ONNX 모델

### Core ML Models (.mlmodel)
- `watermelon_model.mlmodel` - iOS 배포용 Core ML 모델
- `watermelon_model_quantized.mlmodel` - 양자화된 모델

## 모델 변환 체인

```
PyTorch (.pth) → ONNX (.onnx) → Core ML (.mlmodel)
```

## 모델 메타데이터
각 모델과 함께 저장되는 정보:
- 훈련 설정 (config.yaml)
- 성능 지표 (metrics.json)
- 모델 아키텍처 정보
- 전처리 파라미터

## 크기 제한
- PyTorch: 제한 없음
- ONNX: < 100MB 권장
- Core ML: < 50MB (iOS 배포용) 