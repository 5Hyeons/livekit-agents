"""
Session setup and configuration utilities.
"""

import json
import logging
from typing import Dict, Any, Tuple

from livekit import rtc
from livekit.agents.voice.room_io.room_io import RoomInputOptions, RoomOutputOptions

from user_database import UserDatabase, UserData
from config.language_config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, validate_language

logger = logging.getLogger("session-setup")


class SessionSetup:
    """
    Handles session initialization and configuration.
    
    This class manages:
    - Metadata parsing and validation
    - User data initialization
    - Room IO configuration
    - Language and persona setup
    """
    
    @staticmethod
    def parse_participant_metadata(participant: rtc.RemoteParticipant) -> Tuple[str, str, str, str]:
        """
        Parse and validate participant metadata.
        
        Args:
            participant: Remote participant with metadata
            
        Returns:
            Tuple of (user_language, agent_language, custom_persona, voice_name)
        """
        # Default values
        user_language = DEFAULT_LANGUAGE
        agent_language = DEFAULT_LANGUAGE
        custom_persona = ""
        voice_name = "FEMALE_1"
        
        logger.info(f"Parsing participant metadata: {participant.metadata}")
        
        try:
            if participant.metadata:
                metadata = json.loads(participant.metadata)
                
                # Parse user language
                if "userLanguage" in metadata:
                    detected_language = metadata["userLanguage"]
                    if detected_language in SUPPORTED_LANGUAGES:
                        user_language = detected_language
                        logger.info(f"User language detected: {user_language}")
                    else:
                        logger.info(f"Unsupported user language: {detected_language}, using default: {user_language}")
                
                # Parse agent language (currently same as user language)
                if "agentLanguage" in metadata:
                    detected_agent_language = metadata["agentLanguage"]
                    if detected_agent_language in SUPPORTED_LANGUAGES:
                        agent_language = detected_agent_language
                        logger.info(f"Agent language detected: {agent_language}")
                    else:
                        logger.info(f"Unsupported agent language: {detected_agent_language}, using user language: {user_language}")
                        agent_language = user_language
                else:
                    agent_language = user_language
                
                # Parse custom persona
                if "customPersona" in metadata:
                    custom_persona = metadata["customPersona"]
                    if custom_persona and custom_persona.strip():
                        logger.info(f"Custom persona detected (length: {len(custom_persona)})")
                    else:
                        logger.info("Empty custom persona - using default persona")
                
                # Parse voice name
                if "voiceName" in metadata:
                    voice_name = metadata["voiceName"]
                    if voice_name and voice_name.strip():
                        logger.info(f"Voice name detected: {voice_name}")
                    else:
                        logger.info("Empty voice name - using default voice")
                        
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Metadata parsing error: {e}, using defaults: "
                          f"user_language={user_language}, agent_language={agent_language}")
        
        return user_language, agent_language, custom_persona, voice_name
    
    @staticmethod
    def setup_user_data(
        participant: rtc.RemoteParticipant, 
        user_language: str, 
        metadata: Dict[str, Any] = None
    ) -> Tuple[UserDatabase, UserData]:
        """
        Initialize user data and database.
        
        Args:
            participant: Remote participant
            user_language: Validated user language
            metadata: Parsed metadata dictionary
            
        Returns:
            Tuple of (database, user_data)
        """
        # Create user-specific database
        db = UserDatabase(participant.identity)
        user_data = db.get_or_create_user(participant.identity)
        
        # Update user language if changed
        if user_data.language != user_language:
            db.update_user_language(participant.identity, user_language)
            user_data.language = user_language
            logger.info(f"User language updated: {participant.identity} -> {user_language}")
        
        # Save metadata if provided
        if metadata:
            db.update_user_metadata(participant.identity, metadata)
        
        return db, user_data
    
    @staticmethod
    def create_room_options(participant: rtc.RemoteParticipant) -> Tuple[RoomInputOptions, RoomOutputOptions]:
        """
        Create room input and output options.
        
        Args:
            participant: Remote participant
            
        Returns:
            Tuple of (input_options, output_options)
        """
        # Input options - audio only for STT
        room_input_options = RoomInputOptions(
            audio_enabled=True,
            video_enabled=False,
            text_enabled=False,
            participant_identity=participant.identity,
        )
        
        # Output options - animation enabled for face animation
        room_output_options = RoomOutputOptions(
            audio_enabled=False,           # Audio handled by animation stream
            transcription_enabled=False,   # Text transcription disabled
            animation_enabled=True,        # Face animation data enabled
            sync_transcription=False,
        )
        
        logger.info(f"Animation data streaming enabled for: {participant.identity}")
        
        return room_input_options, room_output_options
    
    @staticmethod
    def validate_and_setup_session(participant: rtc.RemoteParticipant) -> Dict[str, Any]:
        """
        Complete session validation and setup.
        
        Args:
            participant: Remote participant
            
        Returns:
            Dictionary containing all setup data
        """
        # Parse metadata
        user_language, agent_language, custom_persona, voice_name = SessionSetup.parse_participant_metadata(participant)
        
        # Validate language
        user_language = validate_language(user_language)
        agent_language = validate_language(agent_language)
        
        # Parse full metadata for storage
        metadata = {}
        try:
            if participant.metadata:
                metadata = json.loads(participant.metadata)
        except (json.JSONDecodeError, Exception):
            pass
        
        # Setup user data
        db, user_data = SessionSetup.setup_user_data(participant, user_language, metadata)
        
        # Create room options
        room_input_options, room_output_options = SessionSetup.create_room_options(participant)
        
        # Log final configuration
        logger.info(f"Final setup - User language: {user_language}, "
                   f"Agent language: {agent_language}, "
                   f"Custom persona: {'configured' if custom_persona else 'default'}, "
                   f"Voice: {voice_name}")
        
        return {
            'user_language': user_language,
            'agent_language': agent_language,
            'custom_persona': custom_persona,
            'voice_name': voice_name,
            'metadata': metadata,
            'db': db,
            'user_data': user_data,
            'room_input_options': room_input_options,
            'room_output_options': room_output_options
        }