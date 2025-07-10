"""
Data Splitter Module
Train/Validation/Test 데이터 분할을 위한 모듈
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import random
import yaml

from .watermelon_dataset import WatermelonDataset


class DataSplitter:
    """
    수박 데이터셋을 Train/Validation/Test로 분할하는 클래스
    
    수박 ID를 기준으로 그룹화하여 분할하므로 같은 수박의 오디오들은
    모두 같은 세트(Train/Val/Test)에 속하게 됩니다.
    """
    
    def __init__(self, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 stratify_by_sweetness: bool = True,
                 sweetness_bins: int = 3,
                 random_seed: int = 42):
        """
        DataSplitter 초기화
        
        Args:
            train_ratio (float): 훈련 데이터 비율 (기본값: 0.7)
            val_ratio (float): 검증 데이터 비율 (기본값: 0.15)
            test_ratio (float): 테스트 데이터 비율 (기본값: 0.15)
            stratify_by_sweetness (bool): 당도별 층화 분할 여부 (기본값: True)
            sweetness_bins (int): 당도 구간 수 (기본값: 3)
            random_seed (int): 랜덤 시드 (기본값: 42)
        """
        # 비율 검증
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"분할 비율의 합이 1.0이 아닙니다: {train_ratio + val_ratio + test_ratio}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify_by_sweetness = stratify_by_sweetness
        self.sweetness_bins = sweetness_bins
        self.random_seed = random_seed
        
        # 랜덤 시드 설정
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def _create_watermelon_summary(self, dataset: WatermelonDataset) -> pd.DataFrame:
        """
        수박별 요약 정보 생성
        
        Args:
            dataset (WatermelonDataset): 수박 데이터셋
            
        Returns:
            pd.DataFrame: 수박별 요약 정보
        """
        watermelon_data = []
        
        for sample in dataset.samples:
            watermelon_data.append({
                'watermelon_id': sample['watermelon_id'],
                'sweetness': sample['sweetness'],
                'folder_name': sample['folder_name']
            })
        
        # 수박별 그룹화
        df = pd.DataFrame(watermelon_data)
        watermelon_summary = df.groupby(['watermelon_id', 'sweetness', 'folder_name']).size().reset_index()
        watermelon_summary.columns = ['watermelon_id', 'sweetness', 'folder_name', 'audio_count']
        
        return watermelon_summary
    
    def _create_sweetness_bins(self, sweetness_values: np.ndarray) -> np.ndarray:
        """
        당도값을 구간별로 분류
        
        Args:
            sweetness_values (np.ndarray): 당도 값들
            
        Returns:
            np.ndarray: 당도 구간 라벨
        """
        # 당도 범위 기반으로 구간 생성
        min_sweetness = sweetness_values.min()
        max_sweetness = sweetness_values.max()
        
        # 구간 경계 생성
        bin_edges = np.linspace(min_sweetness, max_sweetness, self.sweetness_bins + 1)
        
        # 각 수박을 구간에 할당
        bin_labels = np.digitize(sweetness_values, bin_edges[1:-1])
        
        return bin_labels
    
    def split_dataset(self, dataset: WatermelonDataset) -> Tuple[List[int], List[int], List[int]]:
        """
        데이터셋을 Train/Validation/Test로 분할
        
        Args:
            dataset (WatermelonDataset): 분할할 데이터셋
            
        Returns:
            Tuple[List[int], List[int], List[int]]: (train_ids, val_ids, test_ids)
        """
        # 수박별 요약 정보 생성
        watermelon_summary = self._create_watermelon_summary(dataset)
        
        print(f"🍉 데이터 분할 정보")
        print(f"   📊 총 수박 개수: {len(watermelon_summary)}")
        print(f"   🍯 당도 범위: {watermelon_summary['sweetness'].min():.1f} ~ {watermelon_summary['sweetness'].max():.1f}")
        print(f"   📈 분할 비율: Train {self.train_ratio:.1%}, Val {self.val_ratio:.1%}, Test {self.test_ratio:.1%}")
        
        # 수박 ID와 당도 추출  
        watermelon_ids = np.array(watermelon_summary['watermelon_id'].values)
        sweetness_values = np.array(watermelon_summary['sweetness'].values)
        
        if self.stratify_by_sweetness and len(watermelon_summary) >= self.sweetness_bins:
            # 층화 분할 (당도 기준)
            sweetness_bins = self._create_sweetness_bins(sweetness_values)
            
            print(f"   🎯 층화 분할 적용 (당도 기준, {self.sweetness_bins}개 구간)")
            
            # 구간별 분포 출력
            for i in range(self.sweetness_bins):
                count = np.sum(sweetness_bins == i)
                if count > 0:
                    bin_sweetness = sweetness_values[sweetness_bins == i]
                    print(f"      구간 {i+1}: {count}개 (당도 {bin_sweetness.min():.1f}-{bin_sweetness.max():.1f})")
            
            train_ids, temp_ids = self._stratified_split(
                watermelon_ids, sweetness_bins, test_size=(self.val_ratio + self.test_ratio)
            )
            
            # 나머지를 Val/Test로 분할
            # temp_ids에 해당하는 sweetness_bins 추출
            temp_mask = np.isin(watermelon_ids, temp_ids)
            temp_sweetness_bins = sweetness_bins[temp_mask]
            val_test_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
            
            val_ids, test_ids = self._stratified_split(
                np.array(temp_ids), temp_sweetness_bins, test_size=(1 - val_test_ratio)
            )
        
        else:
            # 단순 랜덤 분할
            print(f"   🎲 랜덤 분할 적용")
            
            train_ids, temp_ids = train_test_split(
                watermelon_ids,
                test_size=(self.val_ratio + self.test_ratio),
                random_state=self.random_seed
            )
            
            val_test_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_ids, test_ids = train_test_split(
                temp_ids,
                test_size=(1 - val_test_ratio),
                random_state=self.random_seed
            )
        
        # 결과 요약
        train_ids = train_ids.tolist() if isinstance(train_ids, np.ndarray) else train_ids
        val_ids = val_ids.tolist() if isinstance(val_ids, np.ndarray) else val_ids
        test_ids = test_ids.tolist() if isinstance(test_ids, np.ndarray) else test_ids
        
        print(f"\n✅ 분할 완료:")
        print(f"   🚂 Train: {len(train_ids)}개 수박 ({len(train_ids)/len(watermelon_ids):.1%})")
        print(f"   🔍 Val: {len(val_ids)}개 수박 ({len(val_ids)/len(watermelon_ids):.1%})")
        print(f"   🧪 Test: {len(test_ids)}개 수박 ({len(test_ids)/len(watermelon_ids):.1%})")
        
        # 각 세트의 당도 분포 확인
        self._print_sweetness_distribution(watermelon_summary, train_ids, val_ids, test_ids)
        
        return train_ids, val_ids, test_ids
    
    def _stratified_split(self, 
                         watermelon_ids: np.ndarray, 
                         sweetness_bins: np.ndarray, 
                         test_size: float) -> Tuple[List[int], List[int]]:
        """
        층화 분할 수행
        
        Args:
            watermelon_ids (np.ndarray): 수박 ID 배열
            sweetness_bins (np.ndarray): 당도 구간 라벨
            test_size (float): 테스트 세트 비율
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (train_ids, test_ids)
        """
        try:
            train_ids, test_ids = train_test_split(
                watermelon_ids,
                test_size=test_size,
                stratify=sweetness_bins,
                random_state=self.random_seed
            )
            return train_ids, test_ids
        except ValueError as e:
            # 층화 분할이 불가능한 경우 일반 분할로 대체
            print(f"⚠️ 층화 분할 실패, 랜덤 분할로 대체: {str(e)}")
            return train_test_split(
                watermelon_ids,
                test_size=test_size,
                random_state=self.random_seed
            )
    
    def _print_sweetness_distribution(self, 
                                    watermelon_summary: pd.DataFrame,
                                    train_ids: List[int], 
                                    val_ids: List[int], 
                                    test_ids: List[int]) -> None:
        """분할된 세트별 당도 분포 출력"""
        print(f"\n📊 세트별 당도 분포:")
        
        sets = [
            ("Train", train_ids),
            ("Val", val_ids), 
            ("Test", test_ids)
        ]
        
        for set_name, ids in sets:
            if len(ids) > 0:
                set_sweetness = watermelon_summary[
                    watermelon_summary['watermelon_id'].isin(ids)
                ]['sweetness'].values
                
                print(f"   {set_name:>5}: 평균 {np.mean(set_sweetness):.2f} ± {np.std(set_sweetness):.2f}, "
                      f"범위 {np.min(set_sweetness):.1f}-{np.max(set_sweetness):.1f}")
    
    def create_split_datasets(self, 
                            dataset: WatermelonDataset) -> Tuple[WatermelonDataset, WatermelonDataset, WatermelonDataset]:
        """
        분할된 데이터셋 객체들 생성
        
        Args:
            dataset (WatermelonDataset): 원본 데이터셋
            
        Returns:
            Tuple[WatermelonDataset, WatermelonDataset, WatermelonDataset]: (train_dataset, val_dataset, test_dataset)
        """
        train_ids, val_ids, test_ids = self.split_dataset(dataset)
        
        # 필터링된 데이터셋 생성
        train_dataset = dataset.filter_by_watermelon_ids(train_ids)
        val_dataset = dataset.filter_by_watermelon_ids(val_ids)
        test_dataset = dataset.filter_by_watermelon_ids(test_ids)
        
        return train_dataset, val_dataset, test_dataset
    
    def save_split_info(self, 
                       train_ids: List[int], 
                       val_ids: List[int], 
                       test_ids: List[int],
                       save_path: Union[str, Path]) -> None:
        """
        분할 정보를 파일로 저장
        
        Args:
            train_ids (List[int]): 훈련 세트 수박 ID
            val_ids (List[int]): 검증 세트 수박 ID
            test_ids (List[int]): 테스트 세트 수박 ID
            save_path (Union[str, Path]): 저장 경로
        """
        split_info = {
            'split_config': {
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio,
                'stratify_by_sweetness': self.stratify_by_sweetness,
                'sweetness_bins': self.sweetness_bins,
                'random_seed': self.random_seed
            },
            'splits': {
                'train': sorted(train_ids),
                'val': sorted(val_ids),
                'test': sorted(test_ids)
            },
            'summary': {
                'total_watermelons': len(train_ids) + len(val_ids) + len(test_ids),
                'train_count': len(train_ids),
                'val_count': len(val_ids),
                'test_count': len(test_ids)
            }
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(split_info, f, default_flow_style=False, allow_unicode=True)
        
        print(f"💾 분할 정보 저장됨: {save_path}")
    
    @staticmethod
    def load_split_info(split_file: Union[str, Path]) -> Dict:
        """
        저장된 분할 정보 로드
        
        Args:
            split_file (Union[str, Path]): 분할 정보 파일 경로
            
        Returns:
            Dict: 분할 정보
        """
        split_file = Path(split_file)
        
        if not split_file.exists():
            raise FileNotFoundError(f"분할 정보 파일이 없습니다: {split_file}")
        
        with open(split_file, 'r', encoding='utf-8') as f:
            split_info = yaml.safe_load(f)
        
        return split_info
    
    @staticmethod
    def create_datasets_from_split_file(dataset: WatermelonDataset, 
                                      split_file: Union[str, Path]) -> Tuple[WatermelonDataset, WatermelonDataset, WatermelonDataset]:
        """
        저장된 분할 정보로부터 데이터셋 생성
        
        Args:
            dataset (WatermelonDataset): 원본 데이터셋
            split_file (Union[str, Path]): 분할 정보 파일 경로
            
        Returns:
            Tuple[WatermelonDataset, WatermelonDataset, WatermelonDataset]: (train_dataset, val_dataset, test_dataset)
        """
        split_info = DataSplitter.load_split_info(split_file)
        
        train_ids = split_info['splits']['train']
        val_ids = split_info['splits']['val']
        test_ids = split_info['splits']['test']
        
        print(f"📂 분할 정보 로드됨: {split_file}")
        print(f"   🚂 Train: {len(train_ids)}개 수박")
        print(f"   🔍 Val: {len(val_ids)}개 수박")
        print(f"   🧪 Test: {len(test_ids)}개 수박")
        
        # 필터링된 데이터셋 생성
        train_dataset = dataset.filter_by_watermelon_ids(train_ids)
        val_dataset = dataset.filter_by_watermelon_ids(val_ids)
        test_dataset = dataset.filter_by_watermelon_ids(test_ids)
        
        return train_dataset, val_dataset, test_dataset


def create_data_splits(dataset: WatermelonDataset,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      stratify_by_sweetness: bool = True,
                      random_seed: int = 42,
                      save_split_info: Optional[str] = None) -> Tuple[WatermelonDataset, WatermelonDataset, WatermelonDataset]:
    """
    간편한 데이터 분할 함수
    
    Args:
        dataset (WatermelonDataset): 분할할 데이터셋
        train_ratio (float): 훈련 데이터 비율
        val_ratio (float): 검증 데이터 비율  
        test_ratio (float): 테스트 데이터 비율
        stratify_by_sweetness (bool): 당도별 층화 분할 여부
        random_seed (int): 랜덤 시드
        save_split_info (str, optional): 분할 정보 저장 경로
        
    Returns:
        Tuple[WatermelonDataset, WatermelonDataset, WatermelonDataset]: (train_dataset, val_dataset, test_dataset)
    """
    splitter = DataSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by_sweetness=stratify_by_sweetness,
        random_seed=random_seed
    )
    
    train_dataset, val_dataset, test_dataset = splitter.create_split_datasets(dataset)
    
    # 분할 정보 저장
    if save_split_info:
        train_ids = train_dataset.get_watermelon_ids()
        val_ids = val_dataset.get_watermelon_ids()
        test_ids = test_dataset.get_watermelon_ids()
        
        splitter.save_split_info(train_ids, val_ids, test_ids, save_split_info)
    
    return train_dataset, val_dataset, test_dataset 