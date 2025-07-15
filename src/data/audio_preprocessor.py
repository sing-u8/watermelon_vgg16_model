"""
Audio Preprocessor Module
오디오 파일을 VGG-16 모델에 적합한 멜-스펙트로그램으로 변환하는 모듈
"""

import numpy as np
import librosa
import librosa.display
import soundfile as sf
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union
import yaml
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    오디오 파일을 멜-스펙트로그램으로 변환하는 전처리기
    
    Attributes:
        sample_rate (int): 목표 샘플링 레이트 (기본값: 16000)
        n_mels (int): 멜 필터뱅크 개수 (기본값: 128)
        n_fft (int): FFT 윈도우 크기 (기본값: 2048)
        hop_length (int): hop 길이 (기본값: 512)
        target_size (tuple): 출력 이미지 크기 (기본값: (224, 224))
        normalize (bool): 0-1 범위 정규화 여부 (기본값: True)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        AudioPreprocessor 초기화
        
        Args:
            config_path (str, optional): 설정 파일 경로. None이면 기본값 사용
        """
        if config_path:
            self.load_config(config_path)
        else:
            # 기본 설정값
            self.sample_rate = 16000
            self.n_mels = 128
            self.n_fft = 2048
            self.hop_length = 512
            self.target_size = (224, 224)
            self.normalize = True
            self.duration = None  # 고정 길이 (None이면 원본 길이 사용)
            
        # 멜 필터뱅크 미리 계산
        self.mel_filters = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels
        )
    
    def load_config(self, config_path: str) -> None:
        """
        YAML 설정 파일에서 전처리 파라미터 로드
        
        Args:
            config_path (str): 설정 파일 경로
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        audio_config = config.get('audio', {})
        
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.n_mels = audio_config.get('n_mels', 128)
        self.n_fft = audio_config.get('n_fft', 2048)
        self.hop_length = audio_config.get('hop_length', 512)
        self.target_size = tuple(audio_config.get('target_size', [224, 224]))
        self.normalize = audio_config.get('normalize', True)
        self.duration = audio_config.get('duration', None)
    
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Union[int, float]]:
        """
        오디오 파일을 로드하고 리샘플링
        
        Args:
            file_path (Union[str, Path]): 오디오 파일 경로
            
        Returns:
            Tuple[np.ndarray, int]: (오디오 신호, 샘플링 레이트)
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            ValueError: 지원하지 않는 오디오 형식일 때
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {file_path}")
        
        try:
            # librosa로 오디오 로드 (모노로 변환, 리샘플링)
            y, sr = librosa.load(
                file_path, 
                sr=self.sample_rate, 
                mono=True,
                duration=self.duration
            )
            
            # 무음 구간 제거 (선택사항)
            y, _ = librosa.effects.trim(y, top_db=20)
            
            return y, sr
            
        except Exception as e:
            raise ValueError(f"오디오 파일 로드 실패 ({file_path}): {str(e)}")
    
    def compute_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        오디오 신호에서 멜-스펙트로그램 계산
        
        Args:
            y (np.ndarray): 오디오 신호
            
        Returns:
            np.ndarray: 멜-스펙트로그램 (dB 스케일)
        """
        # STFT 계산
        stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # 파워 스펙트로그램
        power_spectrogram = np.abs(stft) ** 2
        
        # 멜-스펙트로그램 변환
        mel_spectrogram = np.dot(self.mel_filters, power_spectrogram)
        
        # dB 스케일로 변환
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return mel_spectrogram_db
    
    def resize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        스펙트로그램을 목표 크기로 리사이즈
        
        Args:
            spectrogram (np.ndarray): 입력 스펙트로그램
            
        Returns:
            np.ndarray: 리사이즈된 스펙트로그램
        """
        # PIL을 사용한 이미지 리사이즈
        # 값 범위를 0-255로 정규화
        spec_normalized = ((spectrogram - spectrogram.min()) / 
                          (spectrogram.max() - spectrogram.min()) * 255).astype(np.uint8)
        
        # PIL Image로 변환
        pil_image = Image.fromarray(spec_normalized, mode='L')
        
        # 리사이즈 (width, height 순서 주의)
        try:
            # PIL 최신 버전
            resized_pil = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)
        except AttributeError:
            # PIL 구버전 호환성
            resized_pil = pil_image.resize(self.target_size, Image.LANCZOS)
        
        # numpy array로 변환하고 원래 값 범위로 복원
        resized_array = np.array(resized_pil, dtype=np.float32)
        resized = (resized_array / 255.0) * (spectrogram.max() - spectrogram.min()) + spectrogram.min()
        
        return resized
    
    def normalize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        스펙트로그램을 0-1 범위로 정규화
        
        Args:
            spectrogram (np.ndarray): 입력 스펙트로그램
            
        Returns:
            np.ndarray: 정규화된 스펙트로그램
        """
        if self.normalize:
            # Min-Max 정규화
            spec_min = spectrogram.min()
            spec_max = spectrogram.max()
            
            if spec_max > spec_min:
                normalized = (spectrogram - spec_min) / (spec_max - spec_min)
            else:
                normalized = np.zeros_like(spectrogram)
                
            return normalized
        else:
            return spectrogram
    
    def process_audio(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        오디오 파일을 전처리하여 VGG-16 입력 형태로 변환
        
        Args:
            file_path (Union[str, Path]): 오디오 파일 경로
            
        Returns:
            np.ndarray: 전처리된 스펙트로그램 (H, W) 또는 (H, W, 1)
        """
        # 1. 오디오 로드
        y, sr = self.load_audio(file_path)
        
        # 2. 멜-스펙트로그램 계산
        mel_spec = self.compute_mel_spectrogram(y)
        
        # 3. 리사이즈
        mel_spec_resized = self.resize_spectrogram(mel_spec)
        
        # 4. 정규화
        mel_spec_normalized = self.normalize_spectrogram(mel_spec_resized)
        
        return mel_spec_normalized
    
    def process_audio_with_channels(self, file_path: Union[str, Path], 
                                  n_channels: int = 3) -> np.ndarray:
        """
        오디오를 RGB 형태로 변환 (VGG-16 호환성을 위해)
        
        Args:
            file_path (Union[str, Path]): 오디오 파일 경로
            n_channels (int): 출력 채널 수 (기본값: 3)
            
        Returns:
            np.ndarray: (H, W, C) 형태의 전처리된 스펙트로그램
        """
        # 기본 전처리
        mel_spec = self.process_audio(file_path)
        
        if n_channels == 1:
            # 그레이스케일
            return np.expand_dims(mel_spec, axis=-1)
        elif n_channels == 3:
            # RGB로 복제
            return np.stack([mel_spec] * 3, axis=-1)
        else:
            raise ValueError(f"지원하지 않는 채널 수: {n_channels}")
    
    def visualize_spectrogram(self, file_path: Union[str, Path], 
                            save_path: Optional[str] = None) -> None:
        """
        스펙트로그램 시각화
        
        Args:
            file_path (Union[str, Path]): 오디오 파일 경로
            save_path (str, optional): 저장할 이미지 경로
        """
        # 오디오 로드
        y, sr = self.load_audio(file_path)
        
        # 멜-스펙트로그램 계산
        mel_spec = self.compute_mel_spectrogram(y)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        
        # 원본 파형
        plt.subplot(2, 2, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title('Original Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # 원본 멜-스펙트로그램
        plt.subplot(2, 2, 2)
        librosa.display.specshow(
            mel_spec, 
            x_axis='time', 
            y_axis='mel', 
            sr=sr, 
            hop_length=self.hop_length
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Original Mel-Spectrogram')
        
        # 리사이즈된 스펙트로그램
        mel_spec_resized = self.resize_spectrogram(mel_spec)
        plt.subplot(2, 2, 3)
        plt.imshow(mel_spec_resized, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f'Resized Mel-Spectrogram {self.target_size}')
        
        # 정규화된 스펙트로그램
        mel_spec_normalized = self.normalize_spectrogram(mel_spec_resized)
        plt.subplot(2, 2, 4)
        plt.imshow(mel_spec_normalized, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title('Normalized Mel-Spectrogram (0-1)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 스펙트로그램이 저장되었습니다: {save_path}")
        
        plt.show()
    
    def get_audio_info(self, file_path: Union[str, Path]) -> dict:
        """
        오디오 파일 정보 반환
        
        Args:
            file_path (Union[str, Path]): 오디오 파일 경로
            
        Returns:
            dict: 오디오 파일 정보
        """
        y, sr = self.load_audio(file_path)
        mel_spec = self.compute_mel_spectrogram(y)
        
        return {
            'file_path': str(file_path),
            'duration': len(y) / sr,
            'sample_rate': sr,
            'samples': len(y),
            'mel_shape_original': mel_spec.shape,
            'mel_shape_resized': self.target_size,
            'preprocessing_params': {
                'n_mels': self.n_mels,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'target_size': self.target_size,
                'normalize': self.normalize
            }
        }


def create_default_preprocessor() -> AudioPreprocessor:
    """기본 설정으로 AudioPreprocessor 인스턴스 생성"""
    return AudioPreprocessor()


def create_preprocessor_from_config(config_path: str) -> AudioPreprocessor:
    """설정 파일로부터 AudioPreprocessor 인스턴스 생성"""
    return AudioPreprocessor(config_path) 