# 📈 Evaluation & Metrics Module

모델 성능 평가 및 시각화 관련 모듈들을 포함합니다.

## 주요 구성 요소

### 계획된 모듈들:
- `metrics.py` - 평가 메트릭 (MAE, MSE, R², MAPE)
- `visualizer.py` - 결과 시각화
- `evaluator.py` - 종합 평가 클래스
- `model_comparison.py` - 모델 성능 비교

## 평가 메트릭

### 회귀 메트릭
- **MAE**: Mean Absolute Error (평균 절대 오차)
- **MSE**: Mean Squared Error (평균 제곱 오차)
- **R² Score**: 결정계수 (설명력)
- **MAPE**: Mean Absolute Percentage Error (평균 절대 백분율 오차)

## 시각화
- 예측 vs 실제값 scatter plot
- 손실 곡선 그래프
- 멜-스펙트로그램 시각화
- 예측 분포 히스토그램

## 성능 목표
- **MAE** < 0.5 (당도 예측 오차 0.5 이하)
- **R² Score** > 0.8 