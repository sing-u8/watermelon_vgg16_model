# 🔮 Inference Pipeline Module

훈련된 모델을 사용한 추론 파이프라인을 포함합니다.

## 주요 구성 요소

### 계획된 모듈들:
- `predictor.py` - 메인 예측 클래스
- `model_loader.py` - 모델 로딩 (PyTorch, ONNX, Core ML)
- `preprocessing_pipeline.py` - 추론용 전처리
- `postprocessing.py` - 결과 후처리
- `batch_predictor.py` - 배치 예측

## 추론 파이프라인

```
Audio Input → Preprocessing → Model Inference → Postprocessing → Result
     ↓             ↓               ↓               ↓           ↓
  .wav/.m4a    Mel-Spec      Sweetness Value   Confidence   JSON Output
```

## 지원 형식

### 입력
- **오디오 형식**: .wav, .m4a, .mp3
- **배치 처리**: 여러 파일 동시 처리

### 출력
- **당도 예측값**: float (0.0 ~ 15.0)
- **신뢰도**: float (0.0 ~ 1.0)
- **처리 시간**: milliseconds

## 성능 목표
- **추론 시간**: < 1초 (CPU 환경)
- **정확도**: MAE < 0.5 