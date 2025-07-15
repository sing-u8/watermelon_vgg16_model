# 📊 Data Processing Module

이 디렉토리는 수박 오디오 데이터 처리와 관련된 모든 모듈을 포함합니다.

## 주요 구성 요소

### 계획된 모듈들:
- `audio_preprocessor.py` - 오디오 전처리 (샘플링, 멜-스펙트로그램 변환)
- `watermelon_dataset.py` - PyTorch Dataset 클래스
- `augmentation.py` - 데이터 증강 (noise, time shift, pitch shift)
- `data_splitter.py` - 훈련/검증/테스트 데이터 분할

## 데이터 처리 파이프라인

```
Audio Files (.m4a, .mp3, .wav) 
    ↓ (librosa)
Mel-Spectrogram (128 mel bins, 224x224)
    ↓ (normalization)
Normalized Tensor (0-1 range)
    ↓ (augmentation)
Augmented Data
```

## 표준 파라미터
- **샘플링 레이트**: 16000 Hz
- **멜-스펙트로그램**: n_mels=128, fft_size=2048, hop_length=512
- **출력 크기**: 224x224 (VGG-16 호환) 