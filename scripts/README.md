# 📜 Execution Scripts

프로젝트의 주요 작업들을 실행하는 스크립트들을 포함합니다.

## 메인 스크립트들

### 데이터 처리
- `prepare_data.py` - 데이터셋 전처리 및 분할
- `augment_data.py` - 데이터 증강 적용
- `validate_data.py` - 데이터 무결성 검증

### 모델 훈련
- `train_model.py` - 모델 훈련 실행
- `resume_training.py` - 중단된 훈련 재개
- `hyperparameter_tuning.py` - 하이퍼파라미터 튜닝

### 모델 변환
- `convert_to_onnx.py` - PyTorch → ONNX 변환
- `convert_to_coreml.py` - ONNX → Core ML 변환
- `optimize_model.py` - 모델 최적화 (양자화 등)

### 평가 및 추론
- `evaluate_model.py` - 모델 성능 평가
- `predict_single.py` - 단일 파일 예측
- `predict_batch.py` - 배치 예측

## 사용 예시

```bash
# 데이터 준비
python scripts/prepare_data.py --config configs/data.yaml

# 모델 훈련
python scripts/train_model.py --config configs/training.yaml

# 모델 변환
python scripts/convert_to_onnx.py --model models/watermelon_model.pth

# 예측 실행
python scripts/predict_single.py --audio audio_file.wav --model models/watermelon_model.pth
```

## 설정 파일 연동
- 모든 스크립트는 YAML 설정 파일 지원
- 명령행 인자로 설정 오버라이드 가능
- 로깅 및 에러 처리 포함 