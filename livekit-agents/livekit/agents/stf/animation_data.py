from __future__ import annotations

import numpy as np
import struct


class AnimationData:
    """애니메이션 데이터 클래스 - 얼굴 블렌드쉐입 데이터를 나타냅니다."""
    
    def __init__(self, data: bytes, num_features: int = 52, timestamp_us: int = 0, segment_id: str = "", sample_rate: int = 48000) -> None:
        self.data = data  # 직렬화된 블렌드쉐입 데이터 (또는 애니메이션+오디오 페어 데이터)
        self.num_features = num_features  # 블렌드쉐입 특성의 개수
        self.timestamp_us = timestamp_us  # 마이크로초 단위의 타임스탬프
        self.segment_id = segment_id  # 세그먼트 ID
        self.sample_rate = sample_rate
    
    @classmethod
    def from_numpy(cls, arr, timestamp_us: int = 0, segment_id: str = ""):
        """NumPy 배열에서 AnimationData 객체 생성"""
        # NumPy 배열을 직렬화
        data = np.array(arr, dtype=np.float32).tobytes()
        return cls(data=data, num_features=len(arr), timestamp_us=timestamp_us, segment_id=segment_id)
    
    @classmethod
    def from_pair(cls, animation_arr: np.ndarray, audio_samples: np.ndarray, 
                  sample_rate: int, num_channels: int,
                  timestamp_us: int = 0, segment_id: str = ""):
        """애니메이션과 오디오 페어 데이터에서 AnimationData 객체 생성
        
        데이터 구조:
        - 4 bytes: 애니메이션 데이터 길이
        - 4 bytes: 오디오 데이터 길이
        - N bytes: 애니메이션 데이터
        - M bytes: 오디오 데이터
        """
        animation_bytes = np.array(animation_arr, dtype=np.float32).tobytes()
        audio_data = audio_samples.tobytes()
        
        # 헤더 생성 (8 바이트)
        header = struct.pack(
            '<II',  # < = little-endian, I = unsigned int (4 bytes each)
            len(animation_bytes),            # 애니메이션 데이터 길이
            len(audio_data),                 # 오디오 데이터 길이
        )
        
        # 헤더 + 애니메이션 데이터 + 오디오 데이터 결합
        combined_data = header + animation_bytes + audio_data
        
        instance = cls(
            data=combined_data,
            num_features=len(animation_arr),
            timestamp_us=timestamp_us,
            segment_id=segment_id,
            sample_rate=sample_rate
        )
        return instance 