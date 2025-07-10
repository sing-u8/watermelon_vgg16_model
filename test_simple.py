#!/usr/bin/env python3

print("🍉 Watermelon Audio Preprocessor Test")
print("="*40)

try:
    import sys
    import os
    sys.path.append('src')
    
    print("✅ 기본 모듈 import 성공")
    
    # AudioPreprocessor import 테스트
    from src.data.audio_preprocessor import AudioPreprocessor
    print("✅ AudioPreprocessor import 성공")
    
    # 인스턴스 생성 테스트
    preprocessor = AudioPreprocessor()
    print("✅ AudioPreprocessor 인스턴스 생성 성공")
    print(f"   - 샘플링 레이트: {preprocessor.sample_rate} Hz")
    print(f"   - 멜 필터뱅크: {preprocessor.n_mels}개")
    print(f"   - 목표 크기: {preprocessor.target_size}")
    
    # 설정 파일 테스트
    from pathlib import Path
    config_path = Path("configs/data.yaml")
    if config_path.exists():
        preprocessor_config = AudioPreprocessor(str(config_path))
        print("✅ 설정 파일 로딩 성공")
        print(f"   - 설정파일 샘플링 레이트: {preprocessor_config.sample_rate} Hz")
    else:
        print("⚠️ 설정 파일이 없습니다")
    
    # 샘플 오디오 파일 테스트
    data_root = Path("watermelon_sound_data")
    if data_root.exists():
        print(f"✅ 데이터셋 폴더 찾음: {data_root}")
        
        # 첫 번째 오디오 파일 찾기
        sample_file = None
        for folder in data_root.glob("*_*"):
            audios_folder = folder / "audios"
            if audios_folder.exists():
                wav_files = list(audios_folder.glob("*.wav"))
                if wav_files:
                    sample_file = wav_files[0]
                    break
        
        if sample_file:
            print(f"✅ 샘플 오디오 파일: {sample_file}")
            
            # 오디오 정보 가져오기
            info = preprocessor.get_audio_info(sample_file)
            print(f"   📏 길이: {info['duration']:.2f}초")
            print(f"   🔊 샘플링 레이트: {info['sample_rate']} Hz")
            
            # 전처리 실행
            mel_spec = preprocessor.process_audio(sample_file)
            print(f"✅ 전처리 완료: {mel_spec.shape}")
            print(f"   📊 값 범위: {mel_spec.min():.3f} ~ {mel_spec.max():.3f}")
            
        else:
            print("❌ 샘플 오디오 파일을 찾을 수 없습니다")
    else:
        print(f"❌ 데이터셋 폴더를 찾을 수 없습니다: {data_root}")
    
    print("\n🎉 테스트 완료!")
    
except Exception as e:
    print(f"❌ 오류 발생: {str(e)}")
    import traceback
    traceback.print_exc() 