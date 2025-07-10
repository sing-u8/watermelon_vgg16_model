#!/usr/bin/env python3
"""
Watermelon Dataset Analysis Script
수박 오디오 데이터셋 구조 및 특성 분석
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_watermelon_dataset():
    """수박 데이터셋을 분석하고 결과를 출력합니다."""
    
    # 데이터셋 경로
    data_root = Path("watermelon_sound_data")
    
    if not data_root.exists():
        print("❌ 데이터셋 폴더를 찾을 수 없습니다.")
        return
    
    # 수박 폴더 목록 수집
    watermelon_folders = [f for f in data_root.iterdir() if f.is_dir()]
    watermelon_folders.sort()
    
    # 라벨링 규칙 분석
    pattern = r"(\d+)_(\d+\.?\d*)"
    watermelon_data = []
    
    for folder in watermelon_folders:
        match = re.match(pattern, folder.name)
        if match:
            watermelon_id = int(match.group(1))
            sweetness = float(match.group(2))
            
            # 각 폴더의 하위 구조 분석
            subfolders = {}
            for subfolder in ['audio', 'audios', 'chu', 'picture']:
                subfolder_path = folder / subfolder
                if subfolder_path.exists():
                    files = list(subfolder_path.glob('*'))
                    subfolders[subfolder] = len([f for f in files if f.is_file()])
                else:
                    subfolders[subfolder] = 0
            
            watermelon_data.append({
                'id': watermelon_id,
                'sweetness': sweetness,
                'folder_name': folder.name,
                'folder_path': str(folder),
                **subfolders
            })
    
    # DataFrame으로 변환
    df = pd.DataFrame(watermelon_data)
    df = df.sort_values('id').reset_index(drop=True)
    
    # 분석 결과 출력
    print("🍉 " + "="*60)
    print("           WATERMELON DATASET ANALYSIS RESULTS")
    print("🍉 " + "="*60)
    print(f"📊 총 수박 개수: {len(df)}개")
    print(f"🍯 당도 범위: {df['sweetness'].min():.1f} ~ {df['sweetness'].max():.1f} Brix")
    print(f"📈 평균 당도: {df['sweetness'].mean():.2f} ± {df['sweetness'].std():.2f} Brix")
    
    # 오디오 파일 총 개수
    total_wav_files = sum(df['audios'] + df['chu'])
    total_original_files = sum(df['audio'])
    
    print(f"🎵 총 처리된 오디오 파일: {total_wav_files:,}개")
    print(f"📁 총 원본 오디오 파일: {total_original_files}개")
    print(f"🖼️ 총 이미지 파일: {sum(df['picture'])}개")
    
    print("\n📂 폴더별 파일 수 통계:")
    for col in ['audio', 'audios', 'chu', 'picture']:
        print(f"{col:>8}: 평균 {df[col].mean():.1f}개, 범위 {df[col].min()}-{df[col].max()}개")
    
    # 당도별 그룹 분석
    print("\n🍯 당도 구간별 분포:")
    sweetness_bins = pd.cut(df['sweetness'], bins=3, labels=['낮음(8.7-9.6)', '중간(9.7-10.8)', '높음(10.9-12.7)'])
    sweetness_counts = sweetness_bins.value_counts()
    for level, count in sweetness_counts.items():
        print(f"  {level}: {count}개")
    
    print("\n🍯 수박별 당도 정보:")
    print(df[['id', 'sweetness', 'folder_name']].to_string(index=False))
    
    # 오디오 파일 형식 분석
    print("\n🎵 오디오 파일 형식 분석:")
    audio_formats = defaultdict(int)
    total_audio_files = 0
    
    for folder in watermelon_folders:
        for subfolder in ['audio', 'audios', 'chu']:
            subfolder_path = folder / subfolder
            if subfolder_path.exists():
                for file in subfolder_path.glob('*'):
                    if file.is_file():
                        ext = file.suffix.lower()
                        if ext in ['.wav', '.m4a', '.mp3', '.flac']:
                            audio_formats[ext] += 1
                            total_audio_files += 1
    
    print(f"총 오디오 파일: {total_audio_files:,}개")
    for ext, count in sorted(audio_formats.items()):
        percentage = (count / total_audio_files) * 100
        print(f"  {ext}: {count:,}개 ({percentage:.1f}%)")
    
    print("\n" + "="*68)
    
    return df

if __name__ == "__main__":
    df = analyze_watermelon_dataset() 