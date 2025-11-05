"""Simple session setup and configuration."""

import json
import logging
from typing import Dict, Any

from livekit import rtc
from livekit.agents.voice.room_io.room_io import RoomInputOptions, RoomOutputOptions

# from user_database import UserDatabase
from config.languages import validate_language

logger = logging.getLogger("session-manager")


def parse_metadata(participant: rtc.RemoteParticipant) -> Dict[str, Any]:
    """Parse participant metadata with defaults."""
    defaults = {
        'language': 'ko',
        'docentId': 'None'
    }
    
    if not participant.metadata:
        return defaults
        
    try:
        metadata = json.loads(participant.metadata)
        return {
            'language': metadata.get('language', 'ko'),
            'docentId': metadata.get('docentId', 'None')
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Metadata parsing error: {e}, using defaults")
        return defaults


def create_room_options(participant: rtc.RemoteParticipant):
    """Create room input and output options."""
    room_input = RoomInputOptions(
        audio_enabled=True,
        video_enabled=False,
        text_enabled=True,
        participant_identity=participant.identity,
    )
    
    room_output = RoomOutputOptions(
        audio_enabled=True,          # Audio handled by animation stream
        transcription_enabled=True,  # Text transcription enabled
        animation_enabled=True,       # Face animation data enabled
        sync_transcription=True,
    )
    
    return room_input, room_output


def setup_session(participant: rtc.RemoteParticipant) -> Dict[str, Any]:
    """Complete session setup in one simple function."""
    # Parse metadata
    config = parse_metadata(participant)
    
    # Create room options
    room_input, room_output = create_room_options(participant)
    
    logger.info(f"Session setup complete - Language: {config['language']}, "
               f"Docent ID: {config['docentId']}")
    
    return {
        'language': config['language'],
        'docentId': config['docentId'],
        'room_input_options': room_input,
        'room_output_options': room_output
    }