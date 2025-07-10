# 🧪 Test Suite

프로젝트의 모든 구성 요소에 대한 단위 테스트와 통합 테스트를 포함합니다.

## 테스트 구조

### 단위 테스트
- `test_data.py` - 데이터 로딩 및 전처리 테스트
- `test_model.py` - 모델 아키텍처 및 순전파 테스트
- `test_training.py` - 훈련 파이프라인 테스트
- `test_inference.py` - 추론 파이프라인 테스트
- `test_utils.py` - 유틸리티 함수 테스트

### 통합 테스트
- `test_integration.py` - End-to-end 파이프라인 테스트
- `test_model_conversion.py` - 모델 변환 프로세스 테스트
- `test_api.py` - API 엔드포인트 테스트

### 성능 테스트
- `test_performance.py` - 추론 속도 및 메모리 사용량 테스트
- `test_benchmarks.py` - 모델 성능 벤치마크

## 테스트 실행

```bash
# 모든 테스트 실행
pytest tests/

# 특정 모듈 테스트
pytest tests/test_data.py

# 커버리지 포함 실행
pytest --cov=src tests/
```

## 테스트 데이터
- `fixtures/` - 테스트용 샘플 데이터
- 실제 데이터의 작은 부분집합 사용
- 다양한 케이스 커버 (정상, 에러, 경계값)

## CI/CD 통합
- GitHub Actions과 연동
- 코드 푸시 시 자동 테스트 실행
- 테스트 통과 후 모델 배포 