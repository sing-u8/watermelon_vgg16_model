"""
Data Loader Module
수박 당도 예측을 위한 데이터 로더 모듈
"""

import torch
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Dict, Any, Optional, List
import yaml
from pathlib import Path

# 상대 import 
import sys
from pathlib import Path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from data.watermelon_dataset import WatermelonDataset
from data.data_splitter import create_data_splits
from data.augmentation import RandomAudioAugmentation


class WatermelonDataLoader:
    """
    수박 데이터셋을 위한 데이터 로더 클래스
    """
    
    def __init__(self, 
                 train_dataset: WatermelonDataset,
                 val_dataset: WatermelonDataset,
                 test_dataset: WatermelonDataset,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        """
        WatermelonDataLoader 초기화
        
        Args:
            train_dataset (WatermelonDataset): 훈련 데이터셋
            val_dataset (WatermelonDataset): 검증 데이터셋
            test_dataset (WatermelonDataset): 테스트 데이터셋
            batch_size (int): 배치 크기
            num_workers (int): 데이터 로딩 워커 수
            pin_memory (bool): GPU 메모리 고정 여부
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 데이터 로더 생성
        self.train_loader = self._create_train_loader()
        self.val_loader = self._create_val_loader()
        self.test_loader = self._create_test_loader()
    
    def _create_train_loader(self) -> DataLoader:
        """훈련용 데이터 로더 생성"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 훈련 시 셔플
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # 마지막 불완전한 배치 제거
            persistent_workers=self.num_workers > 0
        )
    
    def _create_val_loader(self) -> DataLoader:
        """검증용 데이터 로더 생성"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 검증 시 순서 유지
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )
    
    def _create_test_loader(self) -> DataLoader:
        """테스트용 데이터 로더 생성"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # 테스트 시 순서 유지
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )
    
    def get_data_info(self) -> Dict[str, Any]:
        """데이터 로더 정보 반환"""
        return {
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'test_size': len(self.test_dataset),
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'train_batches': len(self.train_loader),
            'val_batches': len(self.val_loader),
            'test_batches': len(self.test_loader),
            'pin_memory': self.pin_memory
        }
    
    def print_data_info(self):
        """데이터 로더 정보 출력"""
        info = self.get_data_info()
        print(f"📊 데이터 로더 정보")
        print(f"   🚂 Train: {info['train_size']:,}개 샘플, {info['train_batches']:,}개 배치")
        print(f"   🔍 Val: {info['val_size']:,}개 샘플, {info['val_batches']:,}개 배치")
        print(f"   🧪 Test: {info['test_size']:,}개 샘플, {info['test_batches']:,}개 배치")
        print(f"   📦 배치 크기: {info['batch_size']}")
        print(f"   👷 워커 수: {info['num_workers']}")
        print(f"   📌 메모리 고정: {info['pin_memory']}")


def create_data_loaders(data_path: str,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       pin_memory: bool = True,
                       use_augmentation: bool = True,
                       stratify_by_sweetness: bool = True,
                       random_seed: int = 42,
                       split_file: Optional[str] = None,
                       config_path: Optional[str] = None) -> WatermelonDataLoader:
    """
    데이터 로더 생성 함수
    
    Args:
        data_path (str): 데이터 경로
        train_ratio (float): 훈련 데이터 비율
        val_ratio (float): 검증 데이터 비율
        test_ratio (float): 테스트 데이터 비율
        batch_size (int): 배치 크기
        num_workers (int): 워커 수
        pin_memory (bool): 메모리 고정 여부
        use_augmentation (bool): 데이터 증강 사용 여부
        stratify_by_sweetness (bool): 당도별 층화 분할 여부
        random_seed (int): 랜덤 시드
        split_file (str, optional): 기존 분할 파일 경로
        config_path (str, optional): 설정 파일 경로
        
    Returns:
        WatermelonDataLoader: 생성된 데이터 로더
    """
    # 설정 파일에서 파라미터 로드
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                data_config = config.get('data', {})
                
                # 설정 값으로 덮어쓰기 (함수 인자가 우선)
                batch_size = data_config.get('batch_size', batch_size)
                num_workers = data_config.get('num_workers', num_workers)
                pin_memory = data_config.get('pin_memory', pin_memory)
    
    print(f"🍉 데이터 로더 생성 시작")
    print(f"   📁 데이터 경로: {data_path}")
    
    # 원본 데이터셋 생성
    dataset = WatermelonDataset(data_path)
    print(f"   📊 전체 데이터: {len(dataset)}개 샘플")
    
    # 데이터 증강 설정
    if use_augmentation:
        augmentation_config = {
            'add_noise': {'snr_db': (15.0, 25.0), 'noise_type': 'white'},
            'time_shift': {'shift_limit': 0.2},
            'pitch_shift': {'step_range': 2.0},
            'volume_scaling': {'scale_range': (0.8, 1.2)},
            'time_stretch': {'rate_range': (0.9, 1.1)},
            'add_reverb': {'room_range': (0.1, 0.3)}
        }
        augmentation = RandomAudioAugmentation(
            augmentation_config=augmentation_config,
            probability=0.3
        )
        # Note: WatermelonDataset needs to support augmentation in __getitem__
        print(f"   🎭 데이터 증강 준비 완료 (데이터셋에서 적용 필요)")
    
    # 데이터 분할
    if split_file and Path(split_file).exists():
        # 기존 분할 파일 사용
        from data.data_splitter import DataSplitter
        train_dataset, val_dataset, test_dataset = DataSplitter.create_datasets_from_split_file(
            dataset, split_file
        )
        print(f"   📂 기존 분할 파일 사용: {split_file}")
    else:
        # 새로운 분할 생성
        train_dataset, val_dataset, test_dataset = create_data_splits(
            dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_by_sweetness=stratify_by_sweetness,
            random_seed=random_seed,
            save_split_info=split_file
        )
    
    # 데이터 로더 생성
    data_loader = WatermelonDataLoader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    data_loader.print_data_info()
    print(f"✅ 데이터 로더 생성 완료")
    
    return data_loader


def create_single_dataloader(dataset: WatermelonDataset,
                           batch_size: int = 32,
                           shuffle: bool = False,
                           num_workers: int = 4,
                           pin_memory: bool = True,
                           drop_last: bool = False) -> DataLoader:
    """
    단일 데이터셋을 위한 데이터 로더 생성
    
    Args:
        dataset (WatermelonDataset): 데이터셋
        batch_size (int): 배치 크기
        shuffle (bool): 셔플 여부
        num_workers (int): 워커 수
        pin_memory (bool): 메모리 고정 여부
        drop_last (bool): 마지막 배치 제거 여부
        
    Returns:
        DataLoader: 생성된 데이터 로더
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )


def get_sample_batch(data_loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    데이터 로더에서 샘플 배치 추출
    
    Args:
        data_loader (DataLoader): 데이터 로더
        device (torch.device): 디바이스
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (입력, 타겟) 텐서
    """
    data_iter = iter(data_loader)
    inputs, targets = next(data_iter)
    
    return inputs.to(device), targets.to(device)


def calculate_dataset_stats(data_loader: DataLoader) -> Dict[str, float]:
    """
    데이터셋 통계 계산
    
    Args:
        data_loader (DataLoader): 데이터 로더
        
    Returns:
        Dict[str, float]: 데이터셋 통계
    """
    print("📈 데이터셋 통계 계산 중...")
    
    all_targets = []
    total_samples = 0
    
    for _, targets in data_loader:
        all_targets.append(targets)
        total_samples += targets.size(0)
    
    all_targets = torch.cat(all_targets, dim=0)
    
    stats = {
        'mean': float(all_targets.mean()),
        'std': float(all_targets.std()),
        'min': float(all_targets.min()),
        'max': float(all_targets.max()),
        'median': float(all_targets.median()),
        'total_samples': total_samples
    }
    
    print(f"   📊 평균: {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"   📏 범위: {stats['min']:.1f} ~ {stats['max']:.1f}")
    print(f"   📍 중앙값: {stats['median']:.2f}")
    print(f"   🔢 총 샘플: {stats['total_samples']:,}")
    
    return stats


if __name__ == "__main__":
    # 데이터 로더 테스트
    print("🧪 데이터 로더 테스트")
    
    # 데이터 경로 설정
    data_path = "../../watermelon_sound_data"
    
    try:
        # 데이터 로더 생성
        data_loader = create_data_loaders(
            data_path=data_path,
            batch_size=16,
            num_workers=2,
            use_augmentation=True
        )
        
        # 샘플 배치 테스트
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 디바이스: {device}")
        
        # 훈련 데이터 샘플
        train_inputs, train_targets = get_sample_batch(data_loader.train_loader, device)
        print(f"🚂 훈련 배치: 입력 {train_inputs.shape}, 타겟 {train_targets.shape}")
        
        # 통계 계산
        train_stats = calculate_dataset_stats(data_loader.train_loader)
        
        print("✅ 데이터 로더 테스트 완료")
        
    except Exception as e:
        print(f"❌ 데이터 로더 테스트 실패: {e}")
        import traceback
        traceback.print_exc() 