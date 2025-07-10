"""
Watermelon Dataset Module
수박 오디오 데이터셋을 위한 PyTorch Dataset 클래스
"""

import os
import re
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Callable
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from .audio_preprocessor import AudioPreprocessor


class WatermelonDataset(Dataset):
    """
    수박 오디오 데이터셋을 위한 PyTorch Dataset 클래스
    
    오디오 파일을 멜-스펙트로그램으로 변환하고 당도값을 라벨로 사용하는 회귀 데이터셋
    
    Attributes:
        data_root (Path): 데이터셋 루트 경로
        preprocessor (AudioPreprocessor): 오디오 전처리기
        samples (List[Dict]): 샘플 정보 리스트
        cache_dir (Optional[Path]): 캐시 디렉토리 경로
        use_cache (bool): 캐시 사용 여부
        transform (Optional[Callable]): 추가 변환 함수
    """
    
    def __init__(self,
                 data_root: Union[str, Path],
                 config_path: Optional[str] = None,
                 audio_folders: Optional[List[str]] = None,
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 verbose: bool = True):
        """
        WatermelonDataset 초기화
        
        Args:
            data_root (Union[str, Path]): 데이터셋 루트 경로
            config_path (str, optional): 전처리 설정 파일 경로
            audio_folders (List[str], optional): 사용할 오디오 폴더 목록 ['audios', 'chu']
            cache_dir (str, optional): 캐시 디렉토리 경로
            use_cache (bool): 캐시 사용 여부 (기본값: True)
            transform (Callable, optional): 추가 데이터 변환 함수
            target_transform (Callable, optional): 타겟 변환 함수
            verbose (bool): 상세 출력 여부 (기본값: True)
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose
        
        # 기본 오디오 폴더 설정
        if audio_folders is None:
            self.audio_folders = ['audios']  # 'audios' 폴더만 사용 (변환된 WAV 파일)
        else:
            self.audio_folders = audio_folders
        
        # 오디오 전처리기 초기화
        if config_path:
            self.preprocessor = AudioPreprocessor(config_path)
        else:
            self.preprocessor = AudioPreprocessor()
        
        # 캐시 설정
        self.use_cache = use_cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = self.data_root / '.cache'
            if use_cache:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터셋 스캔 및 샘플 수집
        self.samples = self._scan_dataset()
        
        if self.verbose:
            self._print_dataset_info()
    
    def _scan_dataset(self) -> List[Dict]:
        """
        데이터셋을 스캔하여 모든 오디오 파일과 라벨 정보 수집
        
        Returns:
            List[Dict]: 샘플 정보 리스트
        """
        samples = []
        pattern = r"(\d+)_(\d+\.?\d*)"
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"데이터셋 경로가 존재하지 않습니다: {self.data_root}")
        
        # 수박 폴더들 탐색
        watermelon_folders = [f for f in self.data_root.iterdir() if f.is_dir()]
        watermelon_folders.sort()
        
        for folder in watermelon_folders:
            match = re.match(pattern, folder.name)
            if not match:
                continue
            
            watermelon_id = int(match.group(1))
            sweetness = float(match.group(2))
            
            # 각 오디오 폴더에서 파일 수집
            for audio_folder in self.audio_folders:
                audio_path = folder / audio_folder
                if not audio_path.exists():
                    continue
                
                # 오디오 파일 찾기
                audio_files = []
                for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
                    audio_files.extend(audio_path.glob(ext))
                
                for audio_file in audio_files:
                    sample_info = {
                        'watermelon_id': watermelon_id,
                        'sweetness': sweetness,
                        'audio_file': str(audio_file),
                        'folder_name': folder.name,
                        'audio_folder': audio_folder,
                        'file_name': audio_file.name
                    }
                    samples.append(sample_info)
        
        if len(samples) == 0:
            raise ValueError(f"데이터셋에서 오디오 파일을 찾을 수 없습니다: {self.data_root}")
        
        return samples
    
    def _print_dataset_info(self) -> None:
        """데이터셋 정보 출력"""
        print("🍉 WatermelonDataset 정보")
        print("="*50)
        print(f"📁 데이터셋 경로: {self.data_root}")
        print(f"🎵 총 오디오 샘플 수: {len(self.samples)}")
        
        # 수박별 샘플 수
        watermelon_counts = {}
        sweetness_values = []
        
        for sample in self.samples:
            wm_id = sample['watermelon_id']
            sweetness = sample['sweetness']
            
            if wm_id not in watermelon_counts:
                watermelon_counts[wm_id] = 0
            watermelon_counts[wm_id] += 1
            sweetness_values.append(sweetness)
        
        print(f"🍉 수박 개수: {len(watermelon_counts)}개")
        print(f"🍯 당도 범위: {min(sweetness_values):.1f} ~ {max(sweetness_values):.1f} Brix")
        print(f"📊 평균 당도: {np.mean(sweetness_values):.2f} ± {np.std(sweetness_values):.2f} Brix")
        
        # 오디오 폴더별 분포
        folder_counts = {}
        for sample in self.samples:
            folder = sample['audio_folder']
            if folder not in folder_counts:
                folder_counts[folder] = 0
            folder_counts[folder] += 1
        
        print(f"📂 오디오 폴더별 분포:")
        for folder, count in folder_counts.items():
            print(f"   {folder}: {count}개")
        
        print(f"💾 캐시 사용: {'예' if self.use_cache else '아니오'}")
        if self.use_cache:
            print(f"📦 캐시 경로: {self.cache_dir}")
    
    def _get_cache_path(self, sample_info: Dict) -> Path:
        """
        샘플의 캐시 파일 경로 생성
        
        Args:
            sample_info (Dict): 샘플 정보
            
        Returns:
            Path: 캐시 파일 경로
        """
        # 파일 경로와 전처리 설정으로 해시 생성
        file_path = sample_info['audio_file']
        config_str = f"{self.preprocessor.sample_rate}_{self.preprocessor.n_mels}_{self.preprocessor.n_fft}_{self.preprocessor.hop_length}_{self.preprocessor.target_size}"
        
        hash_input = f"{file_path}_{config_str}".encode('utf-8')
        file_hash = hashlib.md5(hash_input).hexdigest()
        
        cache_filename = f"{sample_info['watermelon_id']}_{sample_info['file_name']}_{file_hash}.pkl"
        return self.cache_dir / cache_filename
    
    def _load_from_cache(self, cache_path: Path) -> Optional[np.ndarray]:
        """
        캐시에서 전처리된 데이터 로드
        
        Args:
            cache_path (Path): 캐시 파일 경로
            
        Returns:
            Optional[np.ndarray]: 캐시된 스펙트로그램 또는 None
        """
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 캐시 로드 실패: {cache_path} - {str(e)}")
            return None
    
    def _save_to_cache(self, cache_path: Path, data: np.ndarray) -> None:
        """
        전처리된 데이터를 캐시에 저장
        
        Args:
            cache_path (Path): 캐시 파일 경로
            data (np.ndarray): 저장할 데이터
        """
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 캐시 저장 실패: {cache_path} - {str(e)}")
    
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인덱스에 해당하는 샘플 반환
        
        Args:
            idx (int): 샘플 인덱스
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (스펙트로그램, 당도값)
        """
        if idx >= len(self.samples):
            raise IndexError(f"인덱스가 범위를 벗어났습니다: {idx} >= {len(self.samples)}")
        
        sample_info = self.samples[idx]
        audio_file = Path(sample_info['audio_file'])
        sweetness = sample_info['sweetness']
        
        # 캐시 확인 및 로드
        mel_spec = None
        if self.use_cache:
            cache_path = self._get_cache_path(sample_info)
            mel_spec = self._load_from_cache(cache_path)
        
        # 캐시에 없으면 전처리 실행
        if mel_spec is None:
            try:
                # RGB 형태로 전처리 (VGG-16 호환성)
                mel_spec = self.preprocessor.process_audio_with_channels(
                    audio_file, n_channels=3
                )
                
                # 캐시에 저장
                if self.use_cache:
                    self._save_to_cache(cache_path, mel_spec)
                    
            except Exception as e:
                raise RuntimeError(f"오디오 전처리 실패 ({audio_file}): {str(e)}")
        
        # PyTorch Tensor로 변환
        # mel_spec shape: (H, W, C) -> (C, H, W)
        mel_spec_tensor = torch.from_numpy(mel_spec).float().permute(2, 0, 1)
        sweetness_tensor = torch.tensor(sweetness, dtype=torch.float32)
        
        # 추가 변환 적용
        if self.transform:
            mel_spec_tensor = self.transform(mel_spec_tensor)
        
        if self.target_transform:
            sweetness_tensor = self.target_transform(sweetness_tensor)
        
        return mel_spec_tensor, sweetness_tensor
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        인덱스에 해당하는 샘플 정보 반환
        
        Args:
            idx (int): 샘플 인덱스
            
        Returns:
            Dict: 샘플 정보
        """
        if idx >= len(self.samples):
            raise IndexError(f"인덱스가 범위를 벗어났습니다: {idx} >= {len(self.samples)}")
        
        return self.samples[idx].copy()
    
    def get_sweetness_stats(self) -> Dict[str, float]:
        """
        당도 통계 반환
        
        Returns:
            Dict[str, float]: 당도 통계 정보
        """
        sweetness_values = [sample['sweetness'] for sample in self.samples]
        
        return {
            'mean': float(np.mean(sweetness_values)),
            'std': float(np.std(sweetness_values)),
            'min': float(np.min(sweetness_values)),
            'max': float(np.max(sweetness_values)),
            'median': float(np.median(sweetness_values)),
            'count': len(sweetness_values)
        }
    
    def get_watermelon_ids(self) -> List[int]:
        """
        데이터셋에 포함된 수박 ID 목록 반환
        
        Returns:
            List[int]: 수박 ID 목록
        """
        watermelon_ids = list(set(sample['watermelon_id'] for sample in self.samples))
        return sorted(watermelon_ids)
    
    def filter_by_watermelon_ids(self, watermelon_ids: List[int]) -> 'WatermelonDataset':
        """
        특정 수박 ID들만 포함하는 새로운 데이터셋 생성
        
        Args:
            watermelon_ids (List[int]): 포함할 수박 ID 목록
            
        Returns:
            WatermelonDataset: 필터링된 새로운 데이터셋
        """
        # 새로운 인스턴스 생성
        filtered_dataset = WatermelonDataset.__new__(WatermelonDataset)
        
        # 기본 속성 복사
        filtered_dataset.data_root = self.data_root
        filtered_dataset.preprocessor = self.preprocessor
        filtered_dataset.audio_folders = self.audio_folders
        filtered_dataset.cache_dir = self.cache_dir
        filtered_dataset.use_cache = self.use_cache
        filtered_dataset.transform = self.transform
        filtered_dataset.target_transform = self.target_transform
        filtered_dataset.verbose = False  # 필터링된 데이터셋은 조용히
        
        # 샘플 필터링
        filtered_dataset.samples = [
            sample for sample in self.samples
            if sample['watermelon_id'] in watermelon_ids
        ]
        
        return filtered_dataset
    
    def clear_cache(self) -> None:
        """캐시 파일들 삭제"""
        if not self.use_cache or not self.cache_dir.exists():
            return
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 캐시 파일 삭제 실패: {cache_file} - {str(e)}")
        
        if self.verbose:
            print(f"🗑️ 캐시 파일 {len(cache_files)}개 삭제됨")


def create_dataset_from_config(config_path: str, data_root: str, **kwargs) -> WatermelonDataset:
    """
    설정 파일로부터 WatermelonDataset 생성
    
    Args:
        config_path (str): 설정 파일 경로
        data_root (str): 데이터셋 루트 경로
        **kwargs: 추가 Dataset 파라미터
        
    Returns:
        WatermelonDataset: 생성된 데이터셋
    """
    return WatermelonDataset(
        data_root=data_root,
        config_path=config_path,
        **kwargs
    ) 