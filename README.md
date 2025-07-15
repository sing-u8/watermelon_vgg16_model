# 🍉 Watermelon Ripeness & Sweetness Detection ML Project

수박을 친 소리를 기반으로 수박의 당도와 숙성도를 판별하는 머신러닝 모델 개발 프로젝트

## 🎯 프로젝트 목표

- **오디오 신호** → **멜-스펙트로그램** → **CNN 분류** 파이프라인 구축
- **PyTorch 기반 VGG-16 CNN 모델** 개발
- **iOS 앱 배포**를 위한 모델 변환 (PyTorch → ONNX → Core ML)

## 🏗️ 기술 스택

### Core Technologies
- **Framework**: PyTorch
- **Model Architecture**: VGG-16 기반 CNN
- **Audio Processing**: librosa
- **Model Conversion**: ONNX, coremltools
- **Python Version**: Python 3.8+

### 데이터 파이프라인
```
Audio Files (.m4a, .mp3, .wav) 
    ↓ (librosa)
Mel-Spectrogram (128 mel bins, 224x224)
    ↓ (VGG-16 CNN)
Sweetness/Ripeness Prediction (0.0-15.0)
```

## 📁 프로젝트 구조

```
├── src/                    # 소스 코드
│   ├── data/              # 데이터 처리 모듈
│   ├── models/            # VGG-16 모델 아키텍처
│   ├── training/          # 훈련 파이프라인
│   ├── evaluation/        # 평가 및 메트릭
│   ├── utils/             # 유틸리티 함수
│   └── inference/         # 추론 파이프라인
├── configs/               # YAML 설정 파일
├── models/                # 저장된 모델 (.pth, .onnx, .mlmodel)
├── experiments/           # TensorBoard 로그, 실험 결과
├── notebooks/             # Jupyter 노트북 (EDA, 실험)
├── tests/                 # 단위/통합 테스트
├── scripts/               # 실행 스크립트
├── watermelon_sound_data/ # 원본 데이터셋
└── requirements.txt       # 패키지 의존성
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository_url>
cd wm_vgg_model

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터 전처리 및 분할
python scripts/prepare_data.py --config configs/data.yaml
```

### 3. 모델 훈련

```bash
# VGG-16 모델 훈련
python scripts/train_model.py --config configs/training.yaml
```

### 4. 모델 변환 (iOS 배포용)

```bash
# PyTorch → ONNX → Core ML
python scripts/convert_to_onnx.py --model models/watermelon_model.pth
python scripts/convert_to_coreml.py --onnx models/watermelon_model.onnx
```

## 📊 데이터셋

### 구조
- **라벨링 규칙**: `{번호}_{당도값}` (예: `1_10.5`)
- **오디오 형식**: .m4a, .mp3, .wav
- **당도 범위**: 8.7 ~ 12.7 (Brix 단위)
- **총 샘플 수**: 약 1,500+ 오디오 파일

### 전처리 표준
- **샘플링 레이트**: 16000 Hz
- **멜-스펙트로그램**: n_mels=128, fft_size=2048, hop_length=512
- **정규화**: 0-1 범위로 스케일링
- **데이터 증강**: noise, time shift, pitch shift

## 🤖 모델 아키텍처

### VGG-16 기반 회귀 모델
```
Input: Mel-Spectrogram (3, 224, 224)
    ↓
VGG-16 Feature Extractor (pretrained on ImageNet)
    ↓
Adaptive Average Pooling
    ↓
Classifier: FC(4096) → ReLU → Dropout(0.5) → FC(1)
    ↓
Output: Sweetness Value (float, 0.0-15.0)
```

### 훈련 설정
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 32
- **Early Stopping**: patience=10

## 📈 성능 목표

### 모델 성능
- **MAE** < 0.5 (당도 예측 오차 0.5 이하)
- **R² Score** > 0.8
- **추론 시간** < 1초 (CPU 환경)

### 배포 요구사항
- **모델 크기** < 50MB (iOS 앱용)
- **Core ML 호환성** 확인
- **실시간 처리** 가능

## 🔧 개발 도구

### Jupyter 노트북
```bash
# Jupyter Lab 실행
jupyter lab

# 노트북 실행 순서
# 1. notebooks/01_EDA.ipynb - 데이터 분석
# 2. notebooks/02_Preprocessing.ipynb - 전처리 실험
# 3. notebooks/03_Model_Training.ipynb - 모델 훈련
# 4. notebooks/04_Results_Analysis.ipynb - 결과 분석
```

### TensorBoard
```bash
# 훈련 과정 모니터링
tensorboard --logdir experiments/tensorboard
```

### 테스트 실행
```bash
# 전체 테스트 실행
pytest tests/

# 커버리지 포함
pytest --cov=src tests/
```

## 📋 개발 진행 상황

현재 **Phase 1.1: 프로젝트 구조 생성** 완료! ✅

다음 단계: **Phase 1.2: 환경 설정** (requirements.txt, .gitignore)

자세한 진행 상황은 [.cursor/rules/progress-rule.mdc](.cursor/rules/progress-rule.mdc)를 참조하세요.

## 🤝 기여 가이드

### 개발 규칙
- **코드 스타일**: PEP 8 준수
- **타입 힌팅**: 모든 함수에 type hints 적용
- **문서화**: docstring 및 주석 필수
- **테스트**: 새 기능 추가 시 테스트 코드 포함

### Git 워크플로우
```bash
# 기능 브랜치 생성
git checkout -b feature/new-feature

# 개발 후 커밋
git commit -m "feat: add new feature"

# 푸시 및 PR 생성
git push origin feature/new-feature
```

## 📄 라이선스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.

## 📞 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.

---

**수박의 달콤함을 AI로 예측하는 혁신적인 프로젝트! 🍉🤖** 