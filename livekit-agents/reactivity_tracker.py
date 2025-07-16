"""
ReactivityTracker: Clean metrics collection system for measuring voice agent reactivity.

This module provides precise timing measurements for voice agent response time,
measuring each stage from user speech to agent response using actual metrics timestamps.
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# Legacy enum kept for backwards compatibility
class ReactivityStage(Enum):
    """Stages of voice agent reactivity measurement (legacy)."""
    USER_SPEECH_START = "user_speech_start"
    VAD_DETECTION = "vad_detection"
    USER_SPEECH_END = "user_speech_end"
    STT_COMPLETE = "stt_complete"
    LLM_COMPLETE = "llm_complete"
    TTS_COMPLETE = "tts_complete"
    STF_COMPLETE = "stf_complete"
    AGENT_UTTERANCE_START = "agent_utterance_start"


@dataclass
class ReactivityTimings:
    """Complete timing breakdown for a voice agent response cycle."""
    speech_id: Optional[str] = None
    
    # Raw timestamps
    user_speech_start: Optional[float] = None
    vad_detection: Optional[float] = None  
    user_speech_end: Optional[float] = None
    stt_complete: Optional[float] = None
    llm_complete: Optional[float] = None
    tts_complete: Optional[float] = None
    stf_complete: Optional[float] = None
    agent_utterance_start: Optional[float] = None
    
    # Calculated durations (in milliseconds)
    @property
    def vad_detection_time(self) -> Optional[float]:
        """Time from user speech start to VAD detection."""
        if self.user_speech_start and self.vad_detection:
            return (self.vad_detection - self.user_speech_start) * 1000
        return None
    
    @property
    def stt_processing_time(self) -> Optional[float]:
        """Time from user speech end to STT completion."""
        if self.user_speech_end and self.stt_complete:
            return (self.stt_complete - self.user_speech_end) * 1000
        return None
    
    @property
    def llm_inference_time(self) -> Optional[float]:
        """Time from STT completion to LLM response."""
        if self.stt_complete and self.llm_complete:
            return (self.llm_complete - self.stt_complete) * 1000
        return None
    
    @property
    def tts_generation_time(self) -> Optional[float]:
        """Time from LLM completion to TTS completion."""
        if self.llm_complete and self.tts_complete:
            return (self.tts_complete - self.llm_complete) * 1000
        return None
    
    @property
    def stf_processing_time(self) -> Optional[float]:
        """Time from TTS completion to STF completion."""
        if self.tts_complete and self.stf_complete:
            return (self.stf_complete - self.tts_complete) * 1000
        return None
    
    @property
    def e2e_response_time(self) -> Optional[float]:
        """End-to-end response time from user speech end to agent utterance start."""
        if self.user_speech_end and self.agent_utterance_start:
            return (self.agent_utterance_start - self.user_speech_end) * 1000
        return None
    
    def is_complete(self) -> bool:
        """Check if all critical timing points are captured."""
        return all([
            self.user_speech_end,
            self.stt_complete,
            self.llm_complete,
            self.tts_complete,
            self.agent_utterance_start
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "speech_id": self.speech_id,
            "vad_detection_ms": self.vad_detection_time,
            "stt_processing_ms": self.stt_processing_time,
            "llm_inference_ms": self.llm_inference_time,
            "tts_generation_ms": self.tts_generation_time,
            "stf_processing_ms": self.stf_processing_time,
            "e2e_response_ms": self.e2e_response_time,
        }


class ReactivityTracker:
    """
    Tracks timing events across the voice agent pipeline to measure reactivity.
    
    This tracker correlates timing events using speech_id to ensure accurate
    measurements even with concurrent requests.
    
    Usage:
        tracker = ReactivityTracker()
        
        # Record metrics as they come in
        tracker.record_metrics(stt_metrics)
        tracker.record_metrics(llm_metrics)
        # ... etc
        
        # Manual events for user/agent state changes
        tracker.record_user_speech_start()
        tracker.record_agent_utterance_start(speech_id)
    """
    
    def __init__(self) -> None:
        # Track multiple concurrent speech cycles by speech_id
        self._active_cycles: Dict[str, ReactivityTimings] = {}
        self._current_user_cycle: Optional[ReactivityTimings] = None
        self._last_speech_id: Optional[str] = None  # Track the most recent speech_id
        
    def record_user_speech_start(self) -> None:
        """Record when user starts speaking (VAD detection)."""
        timestamp = time.time()
        
        # Complete any previous cycle if it exists
        if self._current_user_cycle and self._current_user_cycle.is_complete():
            self._log_cycle(self._current_user_cycle)
            
        # Start new user cycle
        self._current_user_cycle = ReactivityTimings(
            user_speech_start=timestamp,
            vad_detection=timestamp  # VAD detection happens at same time as speech start
        )
        
    def record_user_speech_end(self) -> None:
        """Record when user stops speaking."""
        timestamp = time.time()
        if self._current_user_cycle:
            self._current_user_cycle.user_speech_end = timestamp
    
    def record_agent_utterance_start(self, speech_id: str) -> None:
        """Record when agent starts speaking."""
        timestamp = time.time()
        
        # Find the matching cycle by speech_id
        if speech_id in self._active_cycles:
            cycle = self._active_cycles[speech_id]
            cycle.agent_utterance_start = timestamp
            
            # Log if complete
            if cycle.is_complete():
                self._log_cycle(cycle)
                del self._active_cycles[speech_id]
        else:
            # If no matching cycle found, just log a warning (don't create partial cycle)
            logger.warning(f"No matching cycle found for speech_id {speech_id}")
    
    def record_metrics(self, metrics_obj: Any) -> None:
        """Record timing from metrics objects (STT, LLM, TTS, STF)."""
        # Import here to avoid circular imports
        from livekit.agents import metrics
        
        speech_id = getattr(metrics_obj, 'speech_id', None)
        timestamp = getattr(metrics_obj, 'timestamp', time.time())
        
        if isinstance(metrics_obj, metrics.STTMetrics):
            # STT metrics don't have speech_id, so we track user speech directly
            self._record_stt_complete(timestamp, speech_id)
        elif isinstance(metrics_obj, metrics.LLMMetrics):
            # LLM metrics have speech_id, track this for later use
            if speech_id:
                self._last_speech_id = speech_id
            self._record_llm_complete(timestamp, speech_id)
        elif isinstance(metrics_obj, metrics.TTSMetrics):
            # TTS metrics have speech_id
            if speech_id:
                self._last_speech_id = speech_id
            self._record_tts_complete(timestamp, speech_id)
        elif hasattr(metrics, 'STFMetrics') and isinstance(metrics_obj, metrics.STFMetrics):
            # STF metrics don't have speech_id, use the last known speech_id
            if not speech_id and self._last_speech_id:
                speech_id = self._last_speech_id
            self._record_stf_complete(timestamp, speech_id)
        # Ignore other metric types (VAD, EOU, etc.)
    
    def _record_stt_complete(self, timestamp: float, speech_id: Optional[str]) -> None:
        """Record STT completion."""
        cycle = self._get_or_create_cycle(speech_id)
        cycle.stt_complete = timestamp
        
        # If we have user speech end from current cycle, use it
        if self._current_user_cycle and self._current_user_cycle.user_speech_end:
            cycle.user_speech_start = self._current_user_cycle.user_speech_start
            cycle.vad_detection = self._current_user_cycle.vad_detection
            cycle.user_speech_end = self._current_user_cycle.user_speech_end
    
    def _record_llm_complete(self, timestamp: float, speech_id: Optional[str]) -> None:
        """Record LLM completion."""
        cycle = self._get_or_create_cycle(speech_id)
        cycle.llm_complete = timestamp
    
    def _record_tts_complete(self, timestamp: float, speech_id: Optional[str]) -> None:
        """Record TTS completion."""
        cycle = self._get_or_create_cycle(speech_id)
        cycle.tts_complete = timestamp
    
    def _record_stf_complete(self, timestamp: float, speech_id: Optional[str]) -> None:
        """Record STF completion."""
        cycle = self._get_or_create_cycle(speech_id)
        cycle.stf_complete = timestamp
    
    def _get_or_create_cycle(self, speech_id: Optional[str]) -> ReactivityTimings:
        """Get existing cycle or create new one for speech_id."""
        if speech_id is None:
            speech_id = "unknown"
            
        if speech_id not in self._active_cycles:
            self._active_cycles[speech_id] = ReactivityTimings(speech_id=speech_id)
            
        return self._active_cycles[speech_id]
    
    def _log_cycle(self, cycle: ReactivityTimings) -> None:
        """Log completed reactivity metrics."""
        if cycle.is_complete():
            metrics = cycle.to_dict()
            
            # Format for clean logging
            log_parts = []
            if metrics["vad_detection_ms"] is not None:
                log_parts.append(f"VAD: {metrics['vad_detection_ms']:.0f}ms")
            if metrics["stt_processing_ms"] is not None:
                log_parts.append(f"STT: {metrics['stt_processing_ms']:.0f}ms")
            if metrics["llm_inference_ms"] is not None:
                log_parts.append(f"LLM: {metrics['llm_inference_ms']:.0f}ms")
            if metrics["tts_generation_ms"] is not None:
                log_parts.append(f"TTS: {metrics['tts_generation_ms']:.0f}ms")
            if metrics["stf_processing_ms"] is not None:
                log_parts.append(f"STF: {metrics['stf_processing_ms']:.0f}ms")
            if metrics["e2e_response_ms"] is not None:
                log_parts.append(f"E2E: {metrics['e2e_response_ms']:.0f}ms")
            
            if log_parts:
                speech_info = f" [{cycle.speech_id}]" if cycle.speech_id else ""
                logger.info(f"Reactivity Metrics{speech_info}: {' | '.join(log_parts)}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current timing metrics from most recent cycle."""
        if self._current_user_cycle:
            return self._current_user_cycle.to_dict()
        elif self._active_cycles:
            # Return most recent active cycle
            latest_cycle = list(self._active_cycles.values())[-1]
            return latest_cycle.to_dict()
        else:
            return {}
    
    def reset(self) -> None:
        """Reset the tracker for a new session."""
        self._active_cycles.clear()
        self._current_user_cycle = None
        self._last_speech_id = None
    
    # Legacy methods for backwards compatibility
    def record_event(self, stage: ReactivityStage, timestamp: Optional[float] = None) -> None:
        """Legacy method for recording events (backwards compatibility)."""
        logger.warning("record_event is deprecated, use record_metrics or specific record methods")
        
        if timestamp is None:
            timestamp = time.time()
            
        if stage == ReactivityStage.USER_SPEECH_START:
            self.record_user_speech_start()
        elif stage == ReactivityStage.USER_SPEECH_END:
            self.record_user_speech_end()
        # Other stages are handled by record_metrics now