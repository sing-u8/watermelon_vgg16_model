# 🛠️ Utility Functions Module

프로젝트 전반에서 사용되는 공통 유틸리티 함수들을 포함합니다.

## 주요 구성 요소

### 계획된 모듈들:
- `config_utils.py` - YAML 설정 파일 로딩/저장
- `file_utils.py` - 파일 입출력 관련 유틸리티
- `audio_utils.py` - 오디오 파일 관련 공통 함수
- `reproducibility.py` - 재현성 보장 (seed 설정)
- `logging_utils.py` - 로깅 설정
- `device_utils.py` - GPU/CPU 디바이스 관리

## 주요 기능

### 설정 관리
- YAML 파일 파싱
- 하이퍼파라미터 검증
- 환경별 설정 오버라이드

### 재현성
- Random seed 고정 (torch, numpy, random)
- 결정적 연산 설정
- 실험 환경 기록

### 파일 관리
- 안전한 파일 입출력
- 디렉토리 자동 생성
- 파일 형식 검증 