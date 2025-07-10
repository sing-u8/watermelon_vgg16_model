#!/usr/bin/env python3
"""
Audio Preprocessor Test Script
오디오 전처리기 기능 테스트
"""

import sys
sys.path.append('../src')

from pathlib import Path
from src.data.audio_preprocessor import AudioPreprocessor, create_preprocessor_from_config
import numpy as np


def test_basic_functionality():
    """기본 기능 테스트"""
    print("🔧 AudioPreprocessor 기본 기능 테스트")
    print("="*50)
    
    # 1. 기본 전처리기 생성
    preprocessor = AudioPreprocessor()
    
    print(f"✅ 전처리기 생성 완료")
    print(f"   - 샘플링 레이트: {preprocessor.sample_rate} Hz")
    print(f"   - 멜 필터뱅크: {preprocessor.n_mels}개")
    print(f"   - FFT 크기: {preprocessor.n_fft}")
    print(f"   - Hop 길이: {preprocessor.hop_length}")
    print(f"   - 목표 크기: {preprocessor.target_size}")
    
    # 2. 샘플 오디오 파일 찾기
    data_root = Path("../watermelon_sound_data")
    sample_files = []
    
    for folder in data_root.glob("*_*"):
        audios_folder = folder / "audios"
        if audios_folder.exists():
            wav_files = list(audios_folder.glob("*.wav"))
            if wav_files:
                sample_files.append(wav_files[0])
                if len(sample_files) >= 3:  # 3개만 테스트
                    break
    
    if not sample_files:
        print("❌ 테스트할 오디오 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n📁 테스트 파일: {len(sample_files)}개")
    for i, file in enumerate(sample_files):
        print(f"   {i+1}. {file.name} ({file.parent.parent.name})")
    
    # 3. 오디오 전처리 테스트
    print(f"\n🎵 오디오 전처리 테스트")
    for i, audio_file in enumerate(sample_files):
        try:
            # 오디오 정보 가져오기
            info = preprocessor.get_audio_info(audio_file)
            print(f"\n파일 {i+1}: {audio_file.name}")
            print(f"   📏 길이: {info['duration']:.2f}초")
            print(f"   📊 샘플 수: {info['samples']:,}개")
            print(f"   🔊 샘플링 레이트: {info['sample_rate']} Hz")
            print(f"   📈 원본 멜-스펙트로그램: {info['mel_shape_original']}")
            print(f"   🎯 리사이즈된 크기: {info['mel_shape_resized']}")
            
            # 전처리 실행
            mel_spec = preprocessor.process_audio(audio_file)
            print(f"   ✅ 전처리 완료: {mel_spec.shape}")
            print(f"   📊 값 범위: {mel_spec.min():.3f} ~ {mel_spec.max():.3f}")
            
            # RGB 변환 테스트
            mel_spec_rgb = preprocessor.process_audio_with_channels(audio_file, n_channels=3)
            print(f"   🌈 RGB 변환: {mel_spec_rgb.shape}")
            
        except Exception as e:
            print(f"   ❌ 오류: {str(e)}")
    
    print(f"\n✅ 기본 기능 테스트 완료!")


def test_config_loading():
    """설정 파일 로딩 테스트"""
    print(f"\n🔧 설정 파일 로딩 테스트")
    print("="*50)
    
    try:
        # 설정 파일로부터 전처리기 생성
        preprocessor = create_preprocessor_from_config("../configs/data.yaml")
        
        print(f"✅ 설정 파일 로딩 완료")
        print(f"   - 샘플링 레이트: {preprocessor.sample_rate} Hz")
        print(f"   - 멜 필터뱅크: {preprocessor.n_mels}개")
        print(f"   - FFT 크기: {preprocessor.n_fft}")
        print(f"   - Hop 길이: {preprocessor.hop_length}")
        print(f"   - 목표 크기: {preprocessor.target_size}")
        print(f"   - 정규화: {preprocessor.normalize}")
        
    except Exception as e:
        print(f"❌ 설정 파일 로딩 실패: {str(e)}")


def test_edge_cases():
    """엣지 케이스 테스트"""
    print(f"\n🔧 엣지 케이스 테스트")
    print("="*50)
    
    preprocessor = AudioPreprocessor()
    
    # 1. 존재하지 않는 파일
    try:
        preprocessor.process_audio("nonexistent_file.wav")
        print("❌ 존재하지 않는 파일 처리가 성공했습니다 (오류)")
    except FileNotFoundError:
        print("✅ 존재하지 않는 파일 처리 - 올바른 예외 발생")
    except Exception as e:
        print(f"⚠️ 예상과 다른 예외: {str(e)}")
    
    # 2. 다양한 채널 수 테스트
    data_root = Path("../watermelon_sound_data")
    sample_file = None
    
    for folder in data_root.glob("*_*"):
        audios_folder = folder / "audios"
        if audios_folder.exists():
            wav_files = list(audios_folder.glob("*.wav"))
            if wav_files:
                sample_file = wav_files[0]
                break
    
    if sample_file:
        try:
            # 1채널 (그레이스케일)
            mel_spec_1ch = preprocessor.process_audio_with_channels(sample_file, n_channels=1)
            print(f"✅ 1채널 변환: {mel_spec_1ch.shape}")
            
            # 3채널 (RGB)
            mel_spec_3ch = preprocessor.process_audio_with_channels(sample_file, n_channels=3)
            print(f"✅ 3채널 변환: {mel_spec_3ch.shape}")
            
            # 지원하지 않는 채널 수
            try:
                preprocessor.process_audio_with_channels(sample_file, n_channels=5)
                print("❌ 지원하지 않는 채널 수 처리가 성공했습니다 (오류)")
            except ValueError:
                print("✅ 지원하지 않는 채널 수 - 올바른 예외 발생")
                
        except Exception as e:
            print(f"❌ 채널 변환 테스트 실패: {str(e)}")
    
    print(f"✅ 엣지 케이스 테스트 완료!")


def test_performance():
    """성능 테스트"""
    print(f"\n🔧 성능 테스트")
    print("="*50)
    
    import time
    
    preprocessor = AudioPreprocessor()
    
    # 샘플 파일들 수집
    data_root = Path("../watermelon_sound_data")
    sample_files = []
    
    for folder in data_root.glob("*_*"):
        audios_folder = folder / "audios"
        if audios_folder.exists():
            wav_files = list(audios_folder.glob("*.wav"))
            sample_files.extend(wav_files[:2])  # 각 폴더에서 2개씩
            if len(sample_files) >= 10:  # 최대 10개
                break
    
    if not sample_files:
        print("❌ 테스트할 오디오 파일이 없습니다.")
        return
    
    print(f"📁 테스트 파일 수: {len(sample_files)}개")
    
    # 전처리 시간 측정
    start_time = time.time()
    processed_count = 0
    
    for audio_file in sample_files:
        try:
            mel_spec = preprocessor.process_audio(audio_file)
            processed_count += 1
        except Exception as e:
            print(f"⚠️ {audio_file.name} 처리 실패: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if processed_count > 0:
        avg_time = total_time / processed_count
        print(f"✅ 성능 테스트 결과:")
        print(f"   📊 처리된 파일: {processed_count}/{len(sample_files)}개")
        print(f"   ⏱️ 총 시간: {total_time:.2f}초")
        print(f"   📈 평균 처리 시간: {avg_time:.3f}초/파일")
        print(f"   🚀 처리 속도: {1/avg_time:.1f} 파일/초")
    else:
        print("❌ 처리된 파일이 없습니다.")


if __name__ == "__main__":
    print("🍉 Audio Preprocessor Test Suite")
    print("="*60)
    
    try:
        test_basic_functionality()
        test_config_loading()
        test_edge_cases()
        test_performance()
        
        print(f"\n🎉 모든 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc() 