# 🍉 EfficientNet 학습 및 Core ML 변환 가이드

이 문서는 새로운 수박 샘플 데이터를 추가한 후, EfficientNet 모델을 처음부터 다시 학습시키고 iOS용 `.mlmodel` 파일까지 만드는 전체 과정을 단계별로 안내합니다.

---

## 📋 전체 진행 순서

### 1단계: 데이터 준비 및 검증 ✅

#### 1-1. 데이터 구조 확인
```bash
python scripts/analyze_dataset.py
```
- 데이터셋 통계 분석
- 오디오 파일 개수 확인
- 당도 분포 시각화
- 문제 파일 검출

#### 1-2. 전처리 테스트
```bash
python scripts/1_test_preprocessor.py
```
- 전처리 파이프라인 검증
- 멜-스펙트로그램 시각화
- 변환된 이미지 품질 확인

---

### 2단계: EfficientNet 모델 훈련 🚀

#### 2-1. 직접 훈련 (권장)
```bash
python scripts/2_train_efficientnet.py
```
- EfficientNet-B0 모델 생성 및 훈련
- 새로운 데이터로 학습
- 체크포인트 자동 저장
- 실시간 성능 모니터링

**결과물:**
```
experiments/direct_efficientnet_YYYYMMDD_HHMMSS/
├── best_mae_epoch_XX.pth      # 최고 성능 모델
├── best_loss_epoch_XX.pth     # 최고 손실 모델
└── training_log.txt           # 훈련 로그
```

#### 2-2. 설정 파일 기반 훈련 (대안)
```bash
python scripts/run_experiment.py --config configs/efficientnet_model.yaml
```
- 하이퍼파라미터 조정 가능
- TensorBoard 로깅
- 자동 체크포인트 관리

---

### 3단계: 모델 성능 평가 📊

#### 3-1. 훈련 결과 분석
```bash
python scripts/test_evaluation_module.py
```
- MAE, MSE, R² 등 메트릭 계산
- 예측 vs 실제 시각화
- 오차 분석 차트 생성

#### 3-2. 실험 결과 비교 (여러 모델이 있는 경우)
```bash
python scripts/compare_experiments.py
```
- 성능 비교 차트
- 최고 모델 선정
- 종합 성능 리포트

---

### 4단계: Core ML 변환 📱

#### 4-1. 자동 변환 (권장)
```bash
python scripts/3_convert_to_coreml.py
```
- 최신 모델 자동 탐지
- PyTorch → ONNX → Core ML 변환
- 변환 성공 여부 검증
- iOS 호환성 테스트

#### 4-2. 수동 변환 (대안)
```bash
python scripts/convert_efficientnet_to_coreml.py
```
- 특정 모델 파일 지정 가능
- 변환 옵션 커스터마이징
- 변환 성능 비교

---

### 5단계: 최종 검증 및 배포 준비 ✅

#### 5-1. 변환된 모델 테스트
```bash
python scripts/test_opencv.py
```
- Core ML 모델 로드 테스트
- 추론 속도 측정
- 예측 정확도 검증

---

## 🏁 최종 결과물

### 훈련된 모델
```
experiments/direct_efficientnet_YYYYMMDD_HHMMSS/
├── best_mae_epoch_XX.pth          # 최고 성능 PyTorch 모델
├── best_loss_epoch_XX.pth         # 최고 손실 PyTorch 모델
└── training_log.txt               # 훈련 로그
```

### 변환된 모델
```
models/converted/
├── efficientnet_fixed.onnx           # ONNX 형식 (범용)
└── efficientnet_fixed_direct.mlmodel # Core ML 형식 (iOS용)
```

### 성능 리포트
```
test_evaluation_plots/
├── prediction_vs_target.png          # 예측 vs 실제
├── error_analysis.png                # 오차 분석
├── metrics_comparison.png            # 메트릭 비교
└── sweetness_range_performance.png   # 당도별 성능
```

---

## 🎯 권장 실행 순서 (요약)

```bash
python scripts/analyze_dataset.py
python scripts/1_test_preprocessor.py
python scripts/2_train_efficientnet.py
python scripts/test_evaluation_module.py
python scripts/3_convert_to_coreml.py
python scripts/test_opencv.py
```

---

## ⚙️ 주요 설정 파일
- configs/efficientnet_model.yaml
- configs/efficientnet_model_fixed.yaml
- configs/training_batch16.yaml, configs/training_batch32.yaml
- configs/model_regularized.yaml

---

이 가이드에 따라 새로운 데이터로 EfficientNet 모델을 학습하고, iOS용 Core ML 모델(.mlmodel)까지 손쉽게 만들 수 있습니다! 🚀 