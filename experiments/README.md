# 🧪 Experiments & Logs

모델 훈련 실험 결과, 로그, 체크포인트를 저장합니다.

## 디렉토리 구조

```
experiments/
├── tensorboard/           # TensorBoard 로그 파일
├── checkpoints/          # 모델 체크포인트 (.pth)
├── results/              # 실험 결과 (metrics, plots)
└── logs/                 # 텍스트 로그 파일
```

## 실험 추적

### TensorBoard
- 훈련/검증 손실 곡선
- 학습률 변화
- 모델 가중치 히스토그램
- 예측 결과 시각화

### 체크포인트
- `best_model.pth` - 최고 성능 모델
- `latest_model.pth` - 최신 체크포인트
- `epoch_XXX.pth` - 특정 에폭 모델

### 결과 파일
- `experiment_config.yaml` - 실험 설정
- `metrics.json` - 성능 지표
- `plots/` - 시각화 결과

## 실험 명명 규칙
- 날짜_시간_실험명 (예: `20240101_120000_vgg16_baseline`)
- 하이퍼파라미터 변화 기록
- Git commit hash 포함 