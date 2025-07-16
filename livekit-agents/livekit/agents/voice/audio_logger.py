"""
오디오 입력 로깅 유틸리티

Unity 클라이언트에서 전송되는 마이크 입력의 오디오 데이터를  
상세히 로깅하여 입력 인식 문제를 디버깅하는 도구입니다.
"""

import logging
import time
import numpy as np
from typing import Dict, Any
from livekit import rtc

logger = logging.getLogger("audio-logger")

class AudioLogger:
    """오디오 입력 로깅 및 통계 관리 클래스"""
    
    def __init__(self, stats_interval: float = 10.0, frame_log_interval: float = 0.2):
        """
        Args:
            stats_interval: 통계 요약 로깅 간격 (초)
            frame_log_interval: 프레임 로깅 간격 (초)
        """
        self.stats_interval = stats_interval
        self.frame_log_interval = frame_log_interval
        self.reset_stats()
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'frame_count': 0,
            'last_frame_time': 0,
            'total_samples': 0,
            'rms_values': [],
            'peak_values': [],
            'silent_frames': 0,
            'last_stats_time': time.time(),
            'last_frame_log_time': 0  # 프레임 로깅 간격 추적
        }
    
    def calculate_audio_level(self, audio_data: bytes) -> tuple[float, float]:
        """오디오 데이터의 RMS와 피크 값을 계산
        
        Args:
            audio_data: 오디오 데이터 (bytes)
            
        Returns:
            tuple[float, float]: (RMS, 피크값)
        """
        try:
            # bytes를 int16 numpy 배열로 변환
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return 0.0, 0.0
            
            # 정규화 (int16 범위: -32768 ~ 32767)
            normalized = audio_array.astype(np.float32) / 32768.0
            
            # RMS 계산
            rms = np.sqrt(np.mean(normalized ** 2))
            
            # 피크 값 계산
            peak = np.max(np.abs(normalized))
            
            return float(rms), float(peak)
        except Exception as e:
            logger.warning(f"오디오 레벨 계산 오류: {e}")
            return 0.0, 0.0
    
    def log_audio_frame(self, frame: rtc.AudioFrame) -> None:
        """오디오 프레임 정보를 로깅
        
        Args:
            frame: LiveKit 오디오 프레임
        """
        current_time = time.time()
        
        # 오디오 레벨 계산
        rms, peak = self.calculate_audio_level(frame.data.tobytes())
        
        # 통계 업데이트
        self.stats['frame_count'] += 1
        self.stats['total_samples'] += frame.samples_per_channel
        self.stats['rms_values'].append(rms)
        self.stats['peak_values'].append(peak)
        
        # 무음 프레임 감지 (RMS < 0.001)
        if rms < 0.001:
            self.stats['silent_frames'] += 1
        
        # 프레임 간격 계산
        frame_interval = 0
        if self.stats['last_frame_time'] > 0:
            frame_interval = current_time - self.stats['last_frame_time']
        
        self.stats['last_frame_time'] = current_time
        
        # 프레임 로깅 간격 확인 후 로깅
        if (current_time - self.stats['last_frame_log_time']) >= self.frame_log_interval:
            logger.info(
                f"[오디오입력] 프레임 #{self.stats['frame_count']}: "
                f"샘플레이트={frame.sample_rate}Hz, 채널={frame.num_channels}, "
                f"샘플={frame.samples_per_channel}, RMS={rms:.4f}, 피크={peak:.4f}, "
                f"간격={frame_interval*1000:.1f}ms"
            )
            self.stats['last_frame_log_time'] = current_time
        
        # 주기적 통계 요약 로깅
        if current_time - self.stats['last_stats_time'] >= self.stats_interval:
            self.log_stats_summary()
            self.stats['last_stats_time'] = current_time
    
    def log_stats_summary(self) -> None:
        """오디오 통계 요약을 로깅"""
        if self.stats['frame_count'] == 0:
            return
        
        # 평균 RMS 및 피크 계산
        avg_rms = np.mean(self.stats['rms_values']) if self.stats['rms_values'] else 0.0
        avg_peak = np.mean(self.stats['peak_values']) if self.stats['peak_values'] else 0.0
        max_peak = np.max(self.stats['peak_values']) if self.stats['peak_values'] else 0.0
        
        # 무음 비율 계산
        silent_ratio = (self.stats['silent_frames'] / self.stats['frame_count']) * 100
        
        # 예상 오디오 지속 시간 (48kHz 기준)
        estimated_duration = self.stats['total_samples'] / 48000.0
        
        logger.info(
            f"[오디오통계] {self.stats_interval}초 요약: 총 {self.stats['frame_count']}프레임, "
            f"평균 RMS={avg_rms:.4f}, 평균 피크={avg_peak:.4f}, 최대 피크={max_peak:.4f}, "
            f"무음 비율={silent_ratio:.1f}%, 예상 지속시간={estimated_duration:.1f}초"
        )
        
        # 통계 초기화 (메모리 관리를 위해)
        self.stats['rms_values'] = self.stats['rms_values'][-100:]  # 최근 100개만 유지
        self.stats['peak_values'] = self.stats['peak_values'][-100:]  # 최근 100개만 유지
        self.stats['silent_frames'] = 0
        self.stats['frame_count'] = 0
        self.stats['total_samples'] = 0
    
    def get_current_stats(self) -> Dict[str, Any]:
        """현재 통계 정보 반환
        
        Returns:
            Dict[str, Any]: 현재 통계 정보
        """
        if self.stats['frame_count'] == 0:
            return {}
        
        avg_rms = np.mean(self.stats['rms_values']) if self.stats['rms_values'] else 0.0
        avg_peak = np.mean(self.stats['peak_values']) if self.stats['peak_values'] else 0.0
        max_peak = np.max(self.stats['peak_values']) if self.stats['peak_values'] else 0.0
        silent_ratio = (self.stats['silent_frames'] / self.stats['frame_count']) * 100
        
        return {
            'frame_count': self.stats['frame_count'],
            'avg_rms': avg_rms,
            'avg_peak': avg_peak,
            'max_peak': max_peak,
            'silent_ratio': silent_ratio,
            'total_samples': self.stats['total_samples']
        }


# 전역 인스턴스 (기존 코드와의 호환성을 위해) - 200ms 간격으로 로깅
_global_audio_logger = AudioLogger(frame_log_interval=0.2)

def log_audio_frame_info(frame: rtc.AudioFrame) -> None:
    """전역 오디오 로거를 사용하여 프레임 정보 로깅 (기존 호환성)"""
    _global_audio_logger.log_audio_frame(frame)

def calculate_audio_level(audio_data: bytes) -> tuple[float, float]:
    """오디오 레벨 계산 (기존 호환성)"""
    return _global_audio_logger.calculate_audio_level(audio_data)

def log_audio_stats_summary() -> None:
    """통계 요약 로깅 (기존 호환성)"""
    _global_audio_logger.log_stats_summary()

def get_audio_stats() -> Dict[str, Any]:
    """현재 오디오 통계 반환"""
    return _global_audio_logger.get_current_stats()