"""
Models Module
수박 당도 예측을 위한 모델 아키텍처 모듈
"""

from .vgg_watermelon import VGGWatermelon, create_vgg_watermelon

__all__ = [
    'VGGWatermelon',
    'create_vgg_watermelon'
]
