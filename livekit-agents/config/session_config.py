"""Session configuration settings."""

from typing import Optional, Dict, Any

# Session Timeouts
USER_INACTIVITY_TIMEOUT_SECONDS: Optional[int] = 60  # None to disable
SESSION_SETUP_TIMEOUT_SECONDS: int = 30
VAD_ACTIVATION_THRESHOLD: float = 0.4

# Room Configuration Defaults
ROOM_INPUT_DEFAULTS: Dict[str, Any] = {
    "audio_enabled": True,
    "video_enabled": False,
    "text_enabled": False,
}

ROOM_OUTPUT_DEFAULTS: Dict[str, Any] = {
    "audio_enabled": False,          # Audio handled by animation stream
    "transcription_enabled": False,  # Text transcription disabled
    "animation_enabled": True,       # Face animation data enabled
    "sync_transcription": False,
}

# Voice Configuration Defaults
DEFAULT_VOICE_NAME: str = "FEMALE_1"

# Agent Configuration
AGENT_PREEMPTIVE_GENERATION: bool = False  # Experimental feature
AGENT_INTERRUPT_MIN_WORDS: int = 0

# Metrics Collection
METRICS_COLLECTION_ENABLED: bool = True
METRICS_COLLECTION_INTERVAL_SECONDS: int = 60