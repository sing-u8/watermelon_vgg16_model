"""
Audio Data Augmentation Module
오디오 데이터 증강을 위한 다양한 기법들
"""

import numpy as np
import librosa
import librosa.effects
from typing import Tuple, Optional, Union, List, Callable, Dict
import random
import torch
from pathlib import Path


class AudioAugmentation:
    """
    오디오 데이터 증강을 위한 클래스
    
    다양한 오디오 증강 기법을 제공하여 모델의 일반화 성능을 향상시킵니다.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 random_seed: Optional[int] = None):
        """
        AudioAugmentation 초기화
        
        Args:
            sample_rate (int): 샘플링 레이트 (기본값: 16000)
            random_seed (int, optional): 랜덤 시드
        """
        self.sample_rate = sample_rate
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def add_noise(self, 
                  audio: np.ndarray, 
                  snr_db: float = 20.0,
                  noise_type: str = 'white') -> np.ndarray:
        """
        오디오에 노이즈 추가
        
        Args:
            audio (np.ndarray): 입력 오디오 신호
            snr_db (float): Signal-to-Noise Ratio (dB) (기본값: 20.0)
            noise_type (str): 노이즈 타입 ('white', 'pink') (기본값: 'white')
            
        Returns:
            np.ndarray: 노이즈가 추가된 오디오 신호
        """
        # 신호 파워 계산
        signal_power = np.mean(audio ** 2)
        
        # SNR에 따른 노이즈 파워 계산
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # 노이즈 생성
        if noise_type == 'white':
            noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        elif noise_type == 'pink':
            # Pink noise 생성 (1/f noise)
            white_noise = np.random.randn(len(audio))
            # 간단한 pink noise 근사 (low-pass filter)
            noise = np.convolve(white_noise, [1, -0.5], mode='same')
            noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
        else:
            raise ValueError(f"지원하지 않는 노이즈 타입: {noise_type}")
        
        # 노이즈 추가
        augmented_audio = audio + noise
        
        return augmented_audio
    
    def time_shift(self, 
                   audio: np.ndarray, 
                   shift_limit: float = 0.2) -> np.ndarray:
        """
        오디오 시간 이동 (Time Shifting)
        
        Args:
            audio (np.ndarray): 입력 오디오 신호
            shift_limit (float): 최대 시간 이동 (초) (기본값: 0.2)
            
        Returns:
            np.ndarray: 시간 이동된 오디오 신호
        """
        # 시간 이동량 계산 (샘플 단위)
        shift_samples = int(shift_limit * self.sample_rate)
        shift_amount = random.randint(-shift_samples, shift_samples)
        
        if shift_amount == 0:
            return audio
        
        # 시간 이동 적용
        if shift_amount > 0:
            # 오른쪽으로 이동 (앞부분에 0 추가)
            shifted_audio = np.concatenate([np.zeros(shift_amount), audio])
            shifted_audio = shifted_audio[:len(audio)]
        else:
            # 왼쪽으로 이동 (뒷부분에 0 추가)
            shift_amount = abs(shift_amount)
            shifted_audio = np.concatenate([audio[shift_amount:], np.zeros(shift_amount)])
        
        return shifted_audio
    
    def pitch_shift(self, 
                    audio: np.ndarray, 
                    n_steps: Optional[float] = None,
                    step_range: float = 2.0) -> np.ndarray:
        """
        피치 시프트 (Pitch Shifting)
        
        Args:
            audio (np.ndarray): 입력 오디오 신호
            n_steps (float, optional): 피치 변경 스텝 (semitones)
            step_range (float): 피치 변경 범위 (기본값: ±2.0 semitones)
            
        Returns:
            np.ndarray: 피치가 변경된 오디오 신호
        """
        if n_steps is None:
            n_steps = random.uniform(-step_range, step_range)
        
        if n_steps == 0:
            return audio
        
        try:
            # librosa를 사용한 피치 시프트
            shifted_audio = librosa.effects.pitch_shift(
                audio, 
                sr=self.sample_rate, 
                n_steps=n_steps
            )
            return shifted_audio
        except Exception as e:
            # 피치 시프트 실패 시 원본 반환
            print(f"⚠️ 피치 시프트 실패: {str(e)}")
            return audio
    
    def volume_scaling(self, 
                       audio: np.ndarray, 
                       scale_factor: Optional[float] = None,
                       scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        볼륨 스케일링 (Volume Scaling)
        
        Args:
            audio (np.ndarray): 입력 오디오 신호
            scale_factor (float, optional): 볼륨 스케일 팩터
            scale_range (Tuple[float, float]): 스케일 범위 (기본값: 0.8-1.2)
            
        Returns:
            np.ndarray: 볼륨이 조정된 오디오 신호
        """
        if scale_factor is None:
            scale_factor = random.uniform(scale_range[0], scale_range[1])
        
        scaled_audio = audio * scale_factor
        
        # 클리핑 방지
        max_val = np.max(np.abs(scaled_audio))
        if max_val > 1.0:
            scaled_audio = scaled_audio / max_val
        
        return scaled_audio
    
    def time_stretch(self, 
                     audio: np.ndarray, 
                     rate: Optional[float] = None,
                     rate_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        시간 스트레칭 (Time Stretching)
        
        Args:
            audio (np.ndarray): 입력 오디오 신호
            rate (float, optional): 스트레치 비율
            rate_range (Tuple[float, float]): 스트레치 범위 (기본값: 0.9-1.1)
            
        Returns:
            np.ndarray: 시간이 조정된 오디오 신호
        """
        if rate is None:
            rate = random.uniform(rate_range[0], rate_range[1])
        
        if rate == 1.0:
            return audio
        
        try:
            # librosa를 사용한 시간 스트레칭
            stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
            
            # 원래 길이로 맞추기
            if len(stretched_audio) > len(audio):
                # 자르기
                stretched_audio = stretched_audio[:len(audio)]
            elif len(stretched_audio) < len(audio):
                # 패딩
                padding = len(audio) - len(stretched_audio)
                stretched_audio = np.pad(stretched_audio, (0, padding), mode='constant')
            
            return stretched_audio
        except Exception as e:
            # 시간 스트레칭 실패 시 원본 반환
            print(f"⚠️ 시간 스트레칭 실패: {str(e)}")
            return audio
    
    def add_reverb(self, 
                   audio: np.ndarray,
                   room_scale: Optional[float] = None,
                   room_range: Tuple[float, float] = (0.1, 0.3)) -> np.ndarray:
        """
        간단한 리버브 효과 추가
        
        Args:
            audio (np.ndarray): 입력 오디오 신호
            room_scale (float, optional): 룸 스케일 (리버브 강도)
            room_range (Tuple[float, float]): 룸 스케일 범위 (기본값: 0.1-0.3)
            
        Returns:
            np.ndarray: 리버브가 추가된 오디오 신호
        """
        if room_scale is None:
            room_scale = random.uniform(room_range[0], room_range[1])
        
        # 간단한 리버브 구현 (지연된 신호 추가)
        delay_samples = int(0.05 * self.sample_rate)  # 50ms 지연
        
        # 지연된 신호 생성
        delayed_audio = np.pad(audio, (delay_samples, 0), mode='constant')[:-delay_samples]
        
        # 원본과 지연된 신호 혼합
        reverb_audio = audio + (delayed_audio * room_scale)
        
        # 정규화
        max_val = np.max(np.abs(reverb_audio))
        if max_val > 1.0:
            reverb_audio = reverb_audio / max_val
        
        return reverb_audio


class RandomAudioAugmentation:
    """
    랜덤 오디오 증강 클래스
    
    여러 증강 기법을 랜덤하게 조합하여 적용합니다.
    """
    
    def __init__(self, 
                 augmentation_config: Optional[Dict] = None,
                 probability: float = 0.5,
                 sample_rate: int = 16000,
                 random_seed: Optional[int] = None):
        """
        RandomAudioAugmentation 초기화
        
        Args:
            augmentation_config (Dict, optional): 증강 설정
            probability (float): 각 증강 기법 적용 확률 (기본값: 0.5)
            sample_rate (int): 샘플링 레이트 (기본값: 16000)
            random_seed (int, optional): 랜덤 시드
        """
        self.probability = probability
        self.augmentation = AudioAugmentation(sample_rate, random_seed)
        
        # 기본 증강 설정
        if augmentation_config is None:
            self.config = {
                'add_noise': {'snr_db': (15.0, 25.0), 'noise_type': 'white'},
                'time_shift': {'shift_limit': 0.2},
                'pitch_shift': {'step_range': 2.0},
                'volume_scaling': {'scale_range': (0.8, 1.2)},
                'time_stretch': {'rate_range': (0.9, 1.1)},
                'add_reverb': {'room_range': (0.1, 0.3)}
            }
        else:
            self.config = augmentation_config
        
        # 증강 함수 매핑
        self.augmentation_functions = {
            'add_noise': self._apply_noise,
            'time_shift': self._apply_time_shift,
            'pitch_shift': self._apply_pitch_shift,
            'volume_scaling': self._apply_volume_scaling,
            'time_stretch': self._apply_time_stretch,
            'add_reverb': self._apply_reverb
        }
    
    def _apply_noise(self, audio: np.ndarray) -> np.ndarray:
        """노이즈 추가 적용"""
        config = self.config['add_noise']
        if isinstance(config['snr_db'], tuple):
            snr_db = random.uniform(config['snr_db'][0], config['snr_db'][1])
        else:
            snr_db = config['snr_db']
        
        return self.augmentation.add_noise(audio, snr_db, config['noise_type'])
    
    def _apply_time_shift(self, audio: np.ndarray) -> np.ndarray:
        """시간 이동 적용"""
        config = self.config['time_shift']
        return self.augmentation.time_shift(audio, config['shift_limit'])
    
    def _apply_pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """피치 시프트 적용"""
        config = self.config['pitch_shift']
        return self.augmentation.pitch_shift(audio, step_range=config['step_range'])
    
    def _apply_volume_scaling(self, audio: np.ndarray) -> np.ndarray:
        """볼륨 스케일링 적용"""
        config = self.config['volume_scaling']
        return self.augmentation.volume_scaling(audio, scale_range=config['scale_range'])
    
    def _apply_time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """시간 스트레칭 적용"""
        config = self.config['time_stretch']
        return self.augmentation.time_stretch(audio, rate_range=config['rate_range'])
    
    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """리버브 적용"""
        config = self.config['add_reverb']
        return self.augmentation.add_reverb(audio, room_range=config['room_range'])
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        랜덤 증강 적용
        
        Args:
            audio (np.ndarray): 입력 오디오 신호
            
        Returns:
            np.ndarray: 증강된 오디오 신호
        """
        augmented_audio = audio.copy()
        
        # 각 증강 기법을 확률적으로 적용
        for aug_name, aug_func in self.augmentation_functions.items():
            if aug_name in self.config and random.random() < self.probability:
                try:
                    augmented_audio = aug_func(augmented_audio)
                except Exception as e:
                    print(f"⚠️ {aug_name} 증강 실패: {str(e)}")
        
        return augmented_audio


class SpectrogramAugmentation:
    """
    스펙트로그램 도메인에서의 증강 기법들
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        SpectrogramAugmentation 초기화
        
        Args:
            random_seed (int, optional): 랜덤 시드
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def freq_mask(self, 
                  spectrogram: np.ndarray, 
                  freq_mask_param: int = 15) -> np.ndarray:
        """
        주파수 마스킹 (SpecAugment)
        
        Args:
            spectrogram (np.ndarray): 입력 스펙트로그램 (H, W) 또는 (H, W, C)
            freq_mask_param (int): 마스킹할 최대 주파수 밴드 수 (기본값: 15)
            
        Returns:
            np.ndarray: 주파수 마스킹이 적용된 스펙트로그램
        """
        spectrogram = spectrogram.copy()
        
        # 2D 또는 3D 스펙트로그램 처리
        if len(spectrogram.shape) == 3:
            h, w, c = spectrogram.shape
            for channel in range(c):
                spectrogram[:, :, channel] = self._apply_freq_mask(
                    spectrogram[:, :, channel], freq_mask_param
                )
        else:
            spectrogram = self._apply_freq_mask(spectrogram, freq_mask_param)
        
        return spectrogram
    
    def _apply_freq_mask(self, spec: np.ndarray, freq_mask_param: int) -> np.ndarray:
        """단일 채널 주파수 마스킹 적용"""
        h, w = spec.shape
        
        # 마스킹할 주파수 밴드 수 결정
        f = random.randint(0, freq_mask_param)
        if f == 0:
            return spec
        
        # 마스킹 시작 위치 결정
        f0 = random.randint(0, h - f)
        
        # 마스킹 적용
        spec[f0:f0+f, :] = 0
        
        return spec
    
    def time_mask(self, 
                  spectrogram: np.ndarray, 
                  time_mask_param: int = 25) -> np.ndarray:
        """
        시간 마스킹 (SpecAugment)
        
        Args:
            spectrogram (np.ndarray): 입력 스펙트로그램 (H, W) 또는 (H, W, C)
            time_mask_param (int): 마스킹할 최대 시간 프레임 수 (기본값: 25)
            
        Returns:
            np.ndarray: 시간 마스킹이 적용된 스펙트로그램
        """
        spectrogram = spectrogram.copy()
        
        # 2D 또는 3D 스펙트로그램 처리
        if len(spectrogram.shape) == 3:
            h, w, c = spectrogram.shape
            for channel in range(c):
                spectrogram[:, :, channel] = self._apply_time_mask(
                    spectrogram[:, :, channel], time_mask_param
                )
        else:
            spectrogram = self._apply_time_mask(spectrogram, time_mask_param)
        
        return spectrogram
    
    def _apply_time_mask(self, spec: np.ndarray, time_mask_param: int) -> np.ndarray:
        """단일 채널 시간 마스킹 적용"""
        h, w = spec.shape
        
        # 마스킹할 시간 프레임 수 결정
        t = random.randint(0, time_mask_param)
        if t == 0:
            return spec
        
        # 마스킹 시작 위치 결정
        t0 = random.randint(0, w - t)
        
        # 마스킹 적용
        spec[:, t0:t0+t] = 0
        
        return spec


def create_augmentation_transform(config: Dict) -> Callable:
    """
    설정에 따른 증강 변환 함수 생성
    
    Args:
        config (Dict): 증강 설정
        
    Returns:
        Callable: 증강 변환 함수
    """
    audio_aug = RandomAudioAugmentation(
        augmentation_config=config.get('audio_augmentation'),
        probability=config.get('probability', 0.5),
        sample_rate=config.get('sample_rate', 16000),
        random_seed=config.get('random_seed')
    )
    
    spec_aug = SpectrogramAugmentation(
        random_seed=config.get('random_seed')
    )
    
    def augmentation_transform(audio_or_spec):
        """증강 변환 함수"""
        if isinstance(audio_or_spec, np.ndarray) and len(audio_or_spec.shape) == 1:
            # 1D 오디오 신호
            return audio_aug(audio_or_spec)
        else:
            # 스펙트로그램 (2D 또는 3D)
            augmented = audio_or_spec
            if config.get('apply_freq_mask', True):
                augmented = spec_aug.freq_mask(augmented)
            if config.get('apply_time_mask', True):
                augmented = spec_aug.time_mask(augmented)
            return augmented
    
    return augmentation_transform 