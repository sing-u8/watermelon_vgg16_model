# 🏋️ Training Pipeline Module

모델 훈련과 관련된 모든 파이프라인을 포함합니다.

## 주요 구성 요소

### 계획된 모듈들:
- `trainer.py` - 메인 훈련 클래스
- `loss_functions.py` - 손실 함수들 (MSE, MAE, Huber)
- `optimizers.py` - 옵티마이저 설정
- `schedulers.py` - 학습률 스케줄러
- `early_stopping.py` - 조기 종료 로직

## 훈련 파이프라인

```
Data Loading → Model Training → Validation → Checkpointing
      ↓              ↓             ↓            ↓
  DataLoader    Loss Calculation  Metrics    Model Save
```

## 주요 설정
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: patience=10 (validation loss 기준)
- **Batch Size**: 32 (기본값)
- **Epochs**: 100 (early stopping으로 조절) 