# 🤖 Model Architecture Module

VGG-16 기반 수박 당도 예측 모델 아키텍처를 정의합니다.

## 주요 구성 요소

### 계획된 모듈들:
- `vgg_watermelon.py` - VGG-16 기반 회귀 모델
- `base_model.py` - 기본 모델 인터페이스
- `model_utils.py` - 모델 관련 유틸리티 함수

## 모델 아키텍처

```
Input: Mel-Spectrogram (3, 224, 224)
    ↓
VGG-16 Feature Extractor (pretrained)
    ↓
Adaptive Average Pooling
    ↓
Classifier: FC(4096) → ReLU → Dropout → FC(1)
    ↓
Output: Sweetness Value (float)
```

## 주요 특징
- **Base Architecture**: VGG-16 (torchvision.models.vgg16)
- **Task**: Regression (당도 예측)
- **Loss Function**: MSE/MAE/Huber Loss
- **Output**: Single continuous value (당도) 