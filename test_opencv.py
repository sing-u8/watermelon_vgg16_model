#!/usr/bin/env python3

try:
    import cv2
    print(f"✅ OpenCV 설치됨: 버전 {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV 설치 안됨: {e}")

try:
    import numpy as np
    test_image = np.ones((100, 100), dtype=np.uint8) * 127
    resized = cv2.resize(test_image, (50, 50))
    print(f"✅ OpenCV 리사이즈 테스트 성공: {resized.shape}")
except Exception as e:
    print(f"❌ OpenCV 테스트 실패: {e}") 