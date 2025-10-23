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
        'user_language': 'ko',
        'agent_language': 'ko',
        'custom_persona': '',
        'voice_name': 'FEMALE_1',
        'voice_speed_offset': 0.0,
        'model_name': 'gpt', 
        'scene_name': 'default_scene'
    }
    
    if not participant.metadata:
        return defaults
        
    try:
        metadata = json.loads(participant.metadata)
        return {
            'user_language': validate_language(metadata.get('userLanguage', 'ko')),
            'agent_language': validate_language(metadata.get('agentLanguage', 'ko')),
            'custom_persona': metadata.get('customPersona', '').strip(),
            'voice_name': metadata.get('voiceName', 'FEMALE_1').strip() or 'FEMALE_1',
            'voice_speed_offset': metadata.get('voiceSpeedOffset', 0.0),
            'model_name': metadata.get('modelName', 'gemini').strip() or 'gemini',
            'scene_name': metadata.get('sceneName', 'default_scene').strip() or 'default_scene'
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Metadata parsing error: {e}, using defaults")
        return defaults


def create_room_options(participant: rtc.RemoteParticipant):
    """Create room input and output options."""
    room_input = RoomInputOptions(
        audio_enabled=True,
        video_enabled=False,
        text_enabled=False,
        participant_identity=participant.identity,
    )
    
    room_output = RoomOutputOptions(
        audio_enabled=True,          # Audio handled by animation stream
        transcription_enabled=False,  # Text transcription disabled
        animation_enabled=True,       # Face animation data enabled
        sync_transcription=False,
    )
    
    return room_input, room_output


def setup_session(participant: rtc.RemoteParticipant) -> Dict[str, Any]:
    """Complete session setup in one simple function."""
    # Parse metadata
    config = parse_metadata(participant)
    
    # Create room options
    room_input, room_output = create_room_options(participant)
    
    logger.info(f"Session setup complete - Language: {config['user_language']}, "
               f"Voice: {config['voice_name']}, "
               f"Voice speed offset: {config['voice_speed_offset']}, "
               f"Model: {config['model_name']}, "
               f"Scene: {config['scene_name']}, "
               f"Custom persona: {'yes' if config['custom_persona'] else 'no'}")
    
    return {
        'user_language': config['user_language'],
        'agent_language': config['agent_language'],
        'custom_persona': config['custom_persona'],
        'voice_name': config['voice_name'],
        'voice_speed_offset': config['voice_speed_offset'],
        'model_name': config['model_name'],
        'scene_name': config['scene_name'],
        'room_input_options': room_input,
        'room_output_options': room_output
    }