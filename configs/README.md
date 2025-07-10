# ⚙️ Configuration Files

프로젝트의 모든 설정 파일들을 YAML 형식으로 관리합니다.

## 계획된 설정 파일들

### 모델 설정
- `model.yaml` - VGG-16 아키텍처 설정
- `training.yaml` - 훈련 하이퍼파라미터
- `data.yaml` - 데이터 전처리 파라미터

### 환경 설정
- `paths.yaml` - 데이터셋 및 모델 경로
- `device.yaml` - GPU/CPU 디바이스 설정

## 설정 구조 예시

```yaml
# model.yaml
model:
  name: "VGG16Watermelon"
  pretrained: true
  num_classes: 1
  dropout: 0.5

# training.yaml
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping:
    patience: 10
    min_delta: 0.001
```

## 사용 방법
- 모든 설정은 `src/utils/config_utils.py`에서 로딩
- 환경별 오버라이드 지원
- 하이퍼파라미터 검증 포함 