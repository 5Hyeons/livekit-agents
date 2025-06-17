from __future__ import annotations

import numpy as np


class AnimationData:
    """애니메이션 데이터 클래스 - 얼굴 블렌드쉐입 데이터를 나타냅니다."""
    
    def __init__(self, data: bytes, num_features: int = 52, timestamp_us: int = 0, segment_id: str = "") -> None:
        self.data = data  # 직렬화된 블렌드쉐입 데이터
        self.num_features = num_features  # 블렌드쉐입 특성의 개수
        self.timestamp_us = timestamp_us  # 마이크로초 단위의 타임스탬프
        self.segment_id = segment_id  # 세그먼트 ID
    
    @classmethod
    def from_numpy(cls, arr, timestamp_us: int = 0, segment_id: str = ""):
        """NumPy 배열에서 AnimationData 객체 생성"""
        # NumPy 배열을 직렬화
        data = np.array(arr, dtype=np.float32).tobytes()
        return cls(data=data, num_features=len(arr), timestamp_us=timestamp_us, segment_id=segment_id) 