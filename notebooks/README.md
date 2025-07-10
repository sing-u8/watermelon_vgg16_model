# 📓 Jupyter Notebooks

데이터 분석, 실험, 프로토타이핑을 위한 Jupyter 노트북들을 포함합니다.

## 계획된 노트북들

### 데이터 분석
- `01_EDA.ipynb` - 탐색적 데이터 분석
  - 데이터셋 구조 분석
  - 당도 분포 시각화
  - 오디오 파일 특성 분석

### 전처리 실험
- `02_Preprocessing.ipynb` - 전처리 과정 실험
  - 멜-스펙트로그램 변환 실험
  - 다양한 파라미터 비교
  - 데이터 증강 효과 분석

### 모델 실험
- `03_Model_Training.ipynb` - 모델 훈련 실험
  - 다양한 아키텍처 비교
  - 하이퍼파라미터 튜닝
  - 훈련 과정 시각화

### 결과 분석
- `04_Results_Analysis.ipynb` - 결과 분석
  - 모델 성능 평가
  - 오류 사례 분석
  - 예측 결과 시각화

## 사용 가이드
- 각 노트북은 독립적으로 실행 가능
- 공통 함수는 `src/` 모듈에서 import
- 실험 결과는 `experiments/` 디렉토리에 저장

## 실행 순서
1. `01_EDA.ipynb` - 데이터 이해
2. `02_Preprocessing.ipynb` - 전처리 파이프라인 설계
3. `03_Model_Training.ipynb` - 모델 개발
4. `04_Results_Analysis.ipynb` - 최종 분석 