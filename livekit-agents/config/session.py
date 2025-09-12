"""Simple session setup and configuration."""

import json
import logging
from typing import Dict, Any

from livekit import rtc
from livekit.agents.voice.room_io.room_io import RoomInputOptions, RoomOutputOptions

from user_database import UserDatabase
from config.language_config import validate_language

logger = logging.getLogger("session")


def parse_metadata(participant: rtc.RemoteParticipant) -> Dict[str, Any]:
    """Parse participant metadata with defaults."""
    defaults = {
        'user_language': 'ko',
        'custom_persona': '',
        'voice_name': 'FEMALE_1',
        'model_name': 'claude', 
        'scene_name': 'default_scene'
    }
    
    if not participant.metadata:
        return defaults
        
    try:
        metadata = json.loads(participant.metadata)
        return {
            'user_language': validate_language(metadata.get('userLanguage', 'ko')),
            'custom_persona': metadata.get('customPersona', '').strip(),
            'voice_name': metadata.get('voiceName', 'FEMALE_1').strip() or 'FEMALE_1',
            'model_name': metadata.get('modelName', 'claude').strip() or 'claude',
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
    
    # Setup user database
    db = UserDatabase(participant.identity)
    user_data = db.get_or_create_user(participant.identity)
    
    # Update user language if changed
    if user_data.language != config['user_language']:
        db.update_user_language(participant.identity, config['user_language'])
        user_data.language = config['user_language']
    
    # Create room options
    room_input, room_output = create_room_options(participant)
    
    logger.info(f"Session setup complete - Language: {config['user_language']}, "
               f"Voice: {config['voice_name']}, "
               f"Model: {config['model_name']}, "
               f"Scene: {config['scene_name']}, "
               f"Custom persona: {'yes' if config['custom_persona'] else 'no'}")
    
    return {
        'user_language': config['user_language'],
        'custom_persona': config['custom_persona'],
        'voice_name': config['voice_name'],
        'model_name': config['model_name'],
        'scene_name': config['scene_name'],
        'db': db,
        'user_data': user_data,
        'room_input_options': room_input,
        'room_output_options': room_output
    }