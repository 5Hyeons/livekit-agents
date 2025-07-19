"""
StreamingReactivityTracker: Voice agent reactivity measurement for streaming pipelines.

This module provides accurate timing measurements focused on streaming latencies:
- LLM Time To First Token (TTFT)
- TTS Time To First Byte (TTFB)  
- STF Time To First Frame (TTFF)
- VAD detection and STT transcription delays from EOU metrics
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StreamingReactivityTimings:
    """Streaming-focused reactivity timing measurements."""
    
    # Core timestamps
    user_speech_end: Optional[float] = None
    agent_utterance_start: Optional[float] = None
    
    # EOU timing components (from EOU metrics)
    vad_detection_delay: Optional[float] = None  # End of utterance delay
    stt_transcription_delay: Optional[float] = None  # Transcription delay
    turn_callback_delay: Optional[float] = None  # on_user_turn_completed delay
    
    # Streaming latencies (from metrics)
    llm_ttft: Optional[float] = None  # Time to first token (ms)
    tts_ttfb: Optional[float] = None  # Time to first byte (ms) 
    stf_ttff: Optional[float] = None  # Time to first frame (ms)
    
    # Reference timing for E2E calculation
    metrics_received_times: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics_received_times is None:
            self.metrics_received_times = {}
    
    @property
    def e2e_response_time(self) -> Optional[float]:
        """E2E response time from user speech end to agent utterance start."""
        if self.user_speech_end and self.agent_utterance_start:
            duration = (self.agent_utterance_start - self.user_speech_end) * 1000
            return max(0, duration)
        return None
    
    @property 
    def total_pipeline_delay(self) -> Optional[float]:
        """Total delay from EOU metrics (VAD + STT + callback)."""
        delays = [self.vad_detection_delay, self.stt_transcription_delay, self.turn_callback_delay]
        if all(d is not None for d in delays):
            return sum(delays) * 1000  # Convert to ms
        return None
        
    def is_basic_complete(self) -> bool:
        """Check if basic streaming metrics are available."""
        return all([
            self.user_speech_end,
            self.llm_ttft is not None,
            self.tts_ttfb is not None
        ])
    
    def is_complete_with_eou(self) -> bool:
        """Check if EOU + streaming metrics are available."""
        return all([
            self.is_basic_complete(),
            self.vad_detection_delay is not None,
            self.stt_transcription_delay is not None
        ])


class StreamingReactivityTracker:
    """
    Tracks voice agent reactivity using streaming-first metrics.
    
    Focuses on meaningful streaming latencies:
    - VAD detection time (from EOU metrics)  
    - STT transcription time (from EOU metrics)
    - LLM time to first token (from LLM metrics.ttft)
    - TTS time to first byte (from TTS metrics.ttfb)
    - STF time to first frame (custom metric)
    """
    
    def __init__(self) -> None:
        self.current = StreamingReactivityTimings()
        self._complete_logged = False  # 완전한 로그 중복 방지 플래그
        
    def record_user_speech_end(self) -> None:
        """Record when user stops speaking."""
        timestamp = time.time()
        # Reset for new cycle
        self.current = StreamingReactivityTimings()
        self.current.user_speech_end = timestamp
        self._complete_logged = False  # 새 사이클 시작 시 플래그 리셋
        logger.debug(f"User speech ended at {timestamp}")
        
    def record_agent_utterance_start(self) -> None:
        """Record when agent starts speaking."""
        timestamp = time.time()
        self.current.agent_utterance_start = timestamp
        logger.debug(f"Agent utterance started at {timestamp}")
        
        # Log E2E if available
        if self.current.e2e_response_time is not None:
            logger.info(f"📈 E2E 응답시간: {self.current.e2e_response_time:.0f}ms")
    
    def record_eou_metrics(self, eou_metrics) -> None:
        """Record EOU (End of Utterance) timing components."""
        # Convert seconds to milliseconds for consistency
        self.current.vad_detection_delay = eou_metrics.end_of_utterance_delay
        self.current.stt_transcription_delay = eou_metrics.transcription_delay 
        self.current.turn_callback_delay = eou_metrics.on_user_turn_completed_delay
        
        logger.debug(f"EOU metrics: VAD={self.current.vad_detection_delay*1000:.0f}ms, "
                    f"STT={self.current.stt_transcription_delay*1000:.0f}ms, "
                    f"Callback={self.current.turn_callback_delay*1000:.0f}ms")
    
    def record_llm_metrics(self, llm_metrics) -> None:
        """Record LLM streaming metrics (TTFT)."""
        # Use TTFT (time to first token) for streaming latency
        self.current.llm_ttft = llm_metrics.ttft * 1000  # Convert to ms
        self.current.metrics_received_times['llm'] = time.time()
        
        logger.debug(f"LLM TTFT: {self.current.llm_ttft:.0f}ms")
    
    def record_tts_metrics(self, tts_metrics) -> None:
        """Record TTS streaming metrics (TTFB).""" 
        # Use TTFB (time to first byte) for streaming latency
        self.current.tts_ttfb = tts_metrics.ttfb * 1000  # Convert to ms
        self.current.metrics_received_times['tts'] = time.time()
        
        logger.debug(f"TTS TTFB: {self.current.tts_ttfb:.0f}ms")
        
        # Log basic streaming metrics immediately when TTS completes
        if self.current.is_basic_complete():
            self._log_streaming_metrics()
    
    def record_stf_metrics(self, stf_metrics) -> None:
        """Record STF streaming metrics."""
        # TTS 패턴 적용으로 실제 TTFF(Time To First Frame) 측정 가능
        if hasattr(stf_metrics, 'ttff') and stf_metrics.ttff > 0:
            self.current.stf_ttff = stf_metrics.ttff * 1000  # Convert to ms
            logger.debug(f"STF TTFF: {self.current.stf_ttff:.0f}ms (streaming)")
        else:
            # 기존 duration 사용 (fallback)
            self.current.stf_ttff = stf_metrics.duration * 1000  # Convert to ms  
            logger.debug(f"STF duration: {self.current.stf_ttff:.0f}ms (legacy)")
        
        self.current.metrics_received_times['stf'] = time.time()
        
        # 완전한 로그를 아직 출력하지 않았고, 조건이 만족되면 출력
        if not self._complete_logged and self.current.is_complete_with_eou():
            self._log_complete_metrics()
            self._complete_logged = True
    
    def record_metrics(self, metrics_obj: Any) -> None:
        """Process metrics objects from the voice pipeline."""
        from livekit.agents import metrics
        
        if isinstance(metrics_obj, metrics.EOUMetrics):
            self.record_eou_metrics(metrics_obj)
        elif isinstance(metrics_obj, metrics.LLMMetrics):
            self.record_llm_metrics(metrics_obj)
        elif isinstance(metrics_obj, metrics.TTSMetrics):
            self.record_tts_metrics(metrics_obj)
        elif isinstance(metrics_obj, metrics.STFMetrics):
            self.record_stf_metrics(metrics_obj)
        # Ignore STT and VAD metrics as they're covered by EOU
    
    def _log_streaming_metrics(self) -> None:
        """Log streaming-focused reactivity metrics."""
        log_parts = []
        
        # Core streaming latencies
        if self.current.llm_ttft is not None:
            log_parts.append(f"LLM: {self.current.llm_ttft:.0f}ms")
        if self.current.tts_ttfb is not None:
            log_parts.append(f"TTS: {self.current.tts_ttfb:.0f}ms")
            
        if log_parts:
            logger.info(f"🚀 스트리밍 반응속도: {' | '.join(log_parts)}")
    
    def _log_complete_metrics(self) -> None:
        """Log complete reactivity breakdown including EOU components."""
        if not self.current.is_complete_with_eou():
            return
            
        log_parts = []
        
        # EOU components (converted to ms)
        if self.current.vad_detection_delay is not None:
            log_parts.append(f"VAD: {self.current.vad_detection_delay*1000:.0f}ms")
        if self.current.stt_transcription_delay is not None:
            log_parts.append(f"STT: {self.current.stt_transcription_delay*1000:.0f}ms")
            
        # Streaming latencies  
        if self.current.llm_ttft is not None:
            log_parts.append(f"LLM: {self.current.llm_ttft:.0f}ms")
        if self.current.tts_ttfb is not None:
            log_parts.append(f"TTS: {self.current.tts_ttfb:.0f}ms")
        if self.current.stf_ttff is not None:
            log_parts.append(f"STF: {self.current.stf_ttff:.0f}ms")
            
        # E2E timing
        if self.current.e2e_response_time is not None:
            log_parts.append(f"E2E: {self.current.e2e_response_time:.0f}ms")
        
        if log_parts:
            logger.info(f"📊 완전한 반응속도: {' | '.join(log_parts)}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Return current reactivity metrics."""
        return {
            "vad_detection_ms": self.current.vad_detection_delay * 1000 if self.current.vad_detection_delay else None,
            "stt_transcription_ms": self.current.stt_transcription_delay * 1000 if self.current.stt_transcription_delay else None,
            "llm_ttft_ms": self.current.llm_ttft,
            "tts_ttfb_ms": self.current.tts_ttfb,
            "stf_ttff_ms": self.current.stf_ttff,
            "e2e_response_ms": self.current.e2e_response_time,
            "total_pipeline_delay_ms": self.current.total_pipeline_delay,
        }
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.current = StreamingReactivityTimings()
        self._complete_logged = False