#!/usr/bin/env python3

print("🍉 WatermelonDataset Test")
print("="*40)

try:
    import sys
    import os
    sys.path.append('src')
    
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    from src.data.watermelon_dataset import WatermelonDataset, create_dataset_from_config
    
    print("✅ WatermelonDataset import 성공")
    
    # 1. 기본 데이터셋 생성 테스트
    print("\n🔧 기본 데이터셋 생성 테스트")
    print("-" * 30)
    
    data_root = "watermelon_sound_data"
    if not Path(data_root).exists():
        print(f"❌ 데이터셋 폴더가 없습니다: {data_root}")
        exit(1)
    
    # 캐시 비활성화로 빠른 테스트
    dataset = WatermelonDataset(
        data_root=data_root,
        use_cache=False,  # 빠른 테스트를 위해 캐시 비활성화
        verbose=True
    )
    
    print(f"\n✅ 데이터셋 생성 성공!")
    print(f"   📊 총 샘플 수: {len(dataset)}")
    print(f"   🍉 수박 ID 목록: {dataset.get_watermelon_ids()}")
    
    # 2. 당도 통계 확인
    print("\n📊 당도 통계")
    print("-" * 30)
    stats = dataset.get_sweetness_stats()
    for key, value in stats.items():
        if key == 'count':
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value:.3f}")
    
    # 3. 샘플 접근 테스트
    print("\n🎵 샘플 접근 테스트")
    print("-" * 30)
    
    if len(dataset) > 0:
        # 첫 번째 샘플
        sample_info = dataset.get_sample_info(0)
        print(f"첫 번째 샘플 정보:")
        print(f"   🍉 수박 ID: {sample_info['watermelon_id']}")
        print(f"   🍯 당도: {sample_info['sweetness']} Brix")
        print(f"   📁 파일: {sample_info['file_name']}")
        print(f"   📂 폴더: {sample_info['audio_folder']}")
        
        # 실제 데이터 로드 테스트 (한 개만)
        print(f"\n🔄 실제 데이터 로드 테스트...")
        try:
            mel_spec, sweetness = dataset[0]
            print(f"✅ 데이터 로드 성공!")
            print(f"   📈 스펙트로그램 shape: {mel_spec.shape}")
            print(f"   📊 스펙트로그램 dtype: {mel_spec.dtype}")
            print(f"   🍯 당도 값: {sweetness.item():.1f}")
            print(f"   📊 스펙트로그램 범위: {mel_spec.min():.3f} ~ {mel_spec.max():.3f}")
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {str(e)}")
    
    # 4. 설정 파일 테스트
    print("\n⚙️ 설정 파일 테스트")
    print("-" * 30)
    
    config_path = "configs/data.yaml"
    if Path(config_path).exists():
        try:
            dataset_config = create_dataset_from_config(
                config_path=config_path,
                data_root=data_root,
                use_cache=False,
                verbose=False
            )
            print(f"✅ 설정 파일로부터 데이터셋 생성 성공")
            print(f"   📊 샘플 수: {len(dataset_config)}")
        except Exception as e:
            print(f"❌ 설정 파일 테스트 실패: {str(e)}")
    else:
        print(f"⚠️ 설정 파일이 없습니다: {config_path}")
    
    # 5. 데이터 로더 테스트
    print("\n🔄 PyTorch DataLoader 테스트")
    print("-" * 30)
    
    try:
        # 작은 배치 크기로 테스트
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=True, 
            num_workers=0  # 멀티프로세싱 비활성화
        )
        
        print(f"✅ DataLoader 생성 성공")
        print(f"   📦 배치 크기: 2")
        print(f"   🔄 총 배치 수: {len(dataloader)}")
        
        # 첫 번째 배치 로드 테스트
        print(f"\n첫 번째 배치 로드 테스트...")
        for batch_idx, (mel_specs, sweetness_values) in enumerate(dataloader):
            print(f"✅ 배치 로드 성공!")
            print(f"   📈 배치 스펙트로그램 shape: {mel_specs.shape}")
            print(f"   🍯 배치 당도 shape: {sweetness_values.shape}")
            print(f"   📊 당도 값들: {sweetness_values.tolist()}")
            break  # 첫 번째 배치만 테스트
            
    except Exception as e:
        print(f"❌ DataLoader 테스트 실패: {str(e)}")
    
    # 6. 필터링 테스트
    print("\n🔍 데이터셋 필터링 테스트")
    print("-" * 30)
    
    try:
        # 처음 3개 수박만 선택
        watermelon_ids = dataset.get_watermelon_ids()[:3]
        filtered_dataset = dataset.filter_by_watermelon_ids(watermelon_ids)
        
        print(f"✅ 필터링 성공!")
        print(f"   🎯 선택된 수박 ID: {watermelon_ids}")
        print(f"   📊 필터링 전 샘플 수: {len(dataset)}")
        print(f"   📊 필터링 후 샘플 수: {len(filtered_dataset)}")
        
    except Exception as e:
        print(f"❌ 필터링 테스트 실패: {str(e)}")
    
    # 7. 에지 케이스 테스트
    print("\n⚠️ 에지 케이스 테스트")
    print("-" * 30)
    
    try:
        # 범위 밖 인덱스 접근
        try:
            dataset[999999]
            print("❌ 범위 밖 인덱스 접근이 성공했습니다 (오류)")
        except IndexError:
            print("✅ 범위 밖 인덱스 - 올바른 예외 발생")
        
        # 빈 수박 ID 목록으로 필터링
        empty_filtered = dataset.filter_by_watermelon_ids([])
        print(f"✅ 빈 필터링 결과: {len(empty_filtered)} 샘플")
        
    except Exception as e:
        print(f"❌ 에지 케이스 테스트 실패: {str(e)}")
    
    print(f"\n🎉 모든 테스트 완료!")
    
except Exception as e:
    print(f"❌ 테스트 실행 중 오류 발생: {str(e)}")
    import traceback
    traceback.print_exc() 