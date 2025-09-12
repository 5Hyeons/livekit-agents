"""
WallmateAgent class for LiveKit voice AI agents with face animation support.
"""

import logging
import os
import wave
from collections.abc import AsyncIterable
from datetime import datetime
import asyncio
import threading

from config import (
    create_instructions,
    is_tts_logging_enabled,
    is_stt_logging_enabled,
    ensure_logging_directories
)
from .model_factory import get_stt, get_tts, get_stf
from .graph_builder import get_langgraph

from livekit.agents import utils
from livekit.agents import stt
from livekit.agents.stt import SpeechEventType
from livekit.agents.llm import function_tool
from livekit.agents.voice.agent import Agent, ModelSettings
from livekit.rtc import AudioFrame

logger = logging.getLogger("wallmate-agent")


class StreamingWAVWriter:
    """Real-time WAV file writer for streaming audio data."""
    
    def __init__(self, filepath: str, sample_rate: int = None, channels: int = None):
        """
        Initialize streaming WAV writer.
        
        Args:
            filepath: Path to write WAV file
            sample_rate: Audio sample rate (None for auto-detection from first frame)
            channels: Number of audio channels (None for auto-detection from first frame)
        """
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.channels = channels
        self.wav_file = None
        self.frames_written = 0
        self.lock = threading.Lock()
        self.is_open = False
        self.is_initialized = False
        
    def open(self):
        """Open the WAV file for writing (headers will be set on first frame)."""
        try:
            with self.lock:
                if not self.is_open:
                    self.wav_file = wave.open(self.filepath, 'wb')
                    self.is_open = True
                    logger.debug(f"Opened streaming WAV file: {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to open WAV file {self.filepath}: {e}")
            
    def write_frame(self, audio_frame: AudioFrame):
        """Write a single audio frame to the WAV file."""
        try:
            with self.lock:
                if self.is_open and self.wav_file:
                    # Initialize WAV format from first frame if not done yet
                    if not self.is_initialized:
                        self.sample_rate = audio_frame.sample_rate
                        self.channels = audio_frame.num_channels
                        
                        self.wav_file.setnchannels(self.channels)
                        self.wav_file.setsampwidth(2)  # 16-bit audio
                        self.wav_file.setframerate(self.sample_rate)
                        self.is_initialized = True
                        
                        logger.info(f"[STT] WAV format initialized: {self.sample_rate}Hz, {self.channels}ch")
                    
                    self.wav_file.writeframes(audio_frame.data)
                    self.frames_written += 1
        except Exception as e:
            logger.error(f"Failed to write audio frame to {self.filepath}: {e}")
            
    def close(self):
        """Close the WAV file."""
        try:
            with self.lock:
                if self.is_open and self.wav_file:
                    self.wav_file.close()
                    self.is_open = False
                    
                    # Calculate duration if we have audio info
                    duration_info = ""
                    if self.is_initialized and self.sample_rate and self.frames_written > 0:
                        # Each frame contains samples_per_channel samples
                        # Duration estimation (rough approximation)
                        duration_sec = self.frames_written / (self.sample_rate / 160)  # Assuming ~160 samples per frame (10ms frames)
                        duration_info = f", ~{duration_sec:.1f}s"
                    
                    logger.info(f"[STT] Closed streaming WAV file: {os.path.basename(self.filepath)} "
                               f"({self.frames_written} frames{duration_info})")
                    
                    if self.is_initialized:
                        logger.info(f"[STT] Audio format: {self.sample_rate}Hz, {self.channels}ch")
        except Exception as e:
            logger.error(f"Failed to close WAV file {self.filepath}: {e}")
            
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class WallmateAgent(Agent):
    """
    Wallmate Agent with multilingual support and performance tracking.

    This agent extends the base Agent class to provide:
    - Face animation capabilities via STF (Speech-To-Face)
    - Multilingual conversation support
    - Performance metrics tracking
    - User data persistence
    - Custom persona support
    """

    def __init__(
        self,
        participant_identity: str,
        agent_identity: str,
        setup_data: dict,
        mongodb_manager,  # MongoDBManager instance
    ):
        """
        Initialize WallmateAgent with MongoDB-based memory.

        Args:
            participant_identity: Identity of the participant
            agent_identity: Identity of the agent
            setup_data: Setup data containing user language, custom persona, voice name, model name, and scene name
            mongodb_manager: MongoDBManager instance for database operations
        """
        self.participant_identity = participant_identity
        self.agent_identity = agent_identity
        self.mongodb_manager = mongodb_manager
        # self.user_language = setup_data['user_language']
        # self.custom_persona = setup_data['custom_persona']
        # self.voice_name = setup_data['voice_name']
        # self.model_name = setup_data['model_name']
        # self.scene_name = setup_data['scene_name']

        # Initialize logging directories for this user
        self.logging_enabled = (is_tts_logging_enabled(self.participant_identity) or 
                               is_stt_logging_enabled(self.participant_identity))
        self.tts_output_dir = None
        self.stt_input_dir = None
        
        if self.logging_enabled:
            logging_dirs = ensure_logging_directories(self.participant_identity)
            self.tts_output_dir = logging_dirs["tts_output"]
            self.stt_input_dir = logging_dirs["stt_input"]
            logger.info(f"Audio logging enabled for user: {self.participant_identity}")
        else:
            self.stt_input_dir = None
            logger.debug(f"Audio logging disabled for user: {self.participant_identity}")

        if setup_data['custom_persona']:
            logger.info(f"Use Custom persona: {setup_data['custom_persona']}")

        # Performance tracking is now handled in event_handlers.py

        # Create base instructions with persona
        base_instructions = create_instructions(
            setup_data['user_language'], custom_persona=setup_data['custom_persona']
        )

        # Create empty chat context (MongoDB Checkpointer will handle history)
        from livekit.agents.llm import ChatContext
        chat_ctx = ChatContext()
        
        logger.info("Using MongoDB Checkpointer for conversation history")

        # Initialize parent Agent with simple config
        super().__init__(
            instructions=base_instructions,
            chat_ctx=chat_ctx,
            stt=get_stt(setup_data['user_language']),
            llm=get_langgraph(
                model_name=setup_data['model_name'], 
                scene_name=setup_data['scene_name'], 
                participant_id=self.participant_identity,
                mongodb_manager=self.mongodb_manager
            ),
            tts=get_tts(setup_data['voice_name']),
            stf=get_stf(),
        )
    # MongoDB automatically handles conversation history through checkpoints
    

    async def on_enter(self):
        """
        Handle agent entry into conversation session.
        Uses userdata to personalize greeting based on stored user profile.
        """
        logger.info(f"WallmateAgent entering session for user: {self.participant_identity}")
        
        # Get user profile from session userdata
        user_profile = self.session.userdata
        user_name = user_profile.get('name')
        
        # Generate personalized system context
        system_context = f"[SYSTEM_CONTEXT: The user just joined. User name is {user_name}. Respond naturally.]"
        logger.info(f"The User just joined: User name is {user_name}")
        
        await self.session.generate_reply(user_input=system_context, allow_interruptions=False)

    def _save_audio_frames_as_wav(self, audio_frames: list[AudioFrame], text_context: str = "", 
                                  audio_type: str = "tts"):
        """
        Save accumulated audio frames as a WAV file for allowed users only.
        
        Args:
            audio_frames: List of audio frames to save
            text_context: Text that was synthesized (for filename)
            audio_type: Type of audio - "tts" for output or "stt" for input
        """
        # Check if logging is enabled for this user and audio type
        output_dir = self.tts_output_dir if audio_type == "tts" else self.stt_input_dir
        if not self.logging_enabled or not output_dir or not audio_frames:
            return
            
        try:
            # Generate unique filename with timestamp and identities
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds precision
            safe_participant = self.participant_identity.replace("/", "_").replace("\\", "_")
            safe_agent = self.agent_identity.replace("/", "_").replace("\\", "_")
            filename = f"{timestamp}_{safe_participant}_{safe_agent}_{audio_type}.wav"
            filepath = os.path.join(output_dir, filename)
            
            # Combine all audio frames
            combined_frame = utils.audio.combine_frames(audio_frames)
            
            # Save as WAV file
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(combined_frame.num_channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(combined_frame.sample_rate)
                wf.writeframes(combined_frame.data)
            
            # Log success with text context
            text_preview = text_context[:50] + "..." if len(text_context) > 50 else text_context
            logger.info(
                f"[TTS SAVED] File: {filename}, Text: \"{text_preview}\", "
                f"Duration: {combined_frame.duration:.2f}s, Frames: {len(audio_frames)}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save TTS audio to {filepath}: {e}")

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[AudioFrame]:
        """
        Override TTS node to log text input and save audio output.
        
        Args:
            text: Async iterable of text segments to be synthesized
            model_settings: Model settings for TTS processing
            
        Returns:
            AsyncIterable[AudioFrame]: Audio frames from TTS synthesis
        """
        # Collect text for context and audio frames for saving
        collected_text_chunks = []
        collected_audio_frames = []
        
        async def enhanced_log_and_forward_text():
            async for text_chunk in text:
                if text_chunk.strip():  # Only process non-empty text chunks
                    collected_text_chunks.append(text_chunk.strip())
                    # Only log text if TTS logging is enabled for this user
                    if self.logging_enabled:
                        logger.info(
                            f"[TTS] Participant: {self.participant_identity}, "
                            f"Agent: {self.agent_identity}, Text: \"{text_chunk.strip()}\""
                        )
                yield text_chunk
        
        # Get audio frames from parent's tts_node
        audio_stream = super().tts_node(enhanced_log_and_forward_text(), model_settings)
        
        async def collect_and_forward_audio():
            async for audio_frame in audio_stream:
                collected_audio_frames.append(audio_frame)
                yield audio_frame
            
            # After all audio frames are processed, save to file (only for allowed users)
            if collected_audio_frames and collected_text_chunks and self.logging_enabled:
                full_text = " ".join(collected_text_chunks)
                self._save_audio_frames_as_wav(collected_audio_frames, full_text, "tts")
        
        return collect_and_forward_audio()

    async def stt_node(
        self, audio: AsyncIterable[AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterable[stt.SpeechEvent | str]:
        """
        Override STT node to save original audio input in real-time.
        
        Args:
            audio: Async iterable of audio frames from user input
            model_settings: Model settings for STT processing
            
        Returns:
            AsyncIterable[stt.SpeechEvent | str]: Speech events from STT processing
        """
        # Initialize streaming WAV writer if STT logging is enabled
        wav_writer = None
        if is_stt_logging_enabled(self.participant_identity) and self.stt_input_dir:
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            safe_participant = self.participant_identity.replace("/", "_").replace("\\", "_")
            safe_agent = self.agent_identity.replace("/", "_").replace("\\", "_")
            filename = f"{timestamp}_{safe_participant}_{safe_agent}_stt_stream.wav"
            filepath = os.path.join(self.stt_input_dir, filename)
            
            # Create writer with auto-detection of sample_rate and channels
            wav_writer = StreamingWAVWriter(filepath)
            wav_writer.open()
            logger.info(f"[STT] Started real-time audio saving: {filename} (auto-detecting format)")
        
        async def enhanced_stream_and_save_audio():
            try:
                async for audio_frame in audio:
                    # Write audio frame immediately to file if logging is enabled
                    if wav_writer:
                        wav_writer.write_frame(audio_frame)
                    
                    yield audio_frame
            finally:
                # Ensure WAV file is closed when audio stream ends
                if wav_writer:
                    wav_writer.close()
        
        # Get speech events from parent's stt_node with real-time audio saving
        speech_stream = super().stt_node(enhanced_stream_and_save_audio(), model_settings)
        
        async def log_and_forward_speech():
            async for speech_event in speech_stream:
                # Log transcription if enabled and it's a transcript event
                if (isinstance(speech_event, stt.SpeechEvent) and 
                    speech_event.type in [SpeechEventType.INTERIM_TRANSCRIPT, 
                                         SpeechEventType.FINAL_TRANSCRIPT] and
                    is_stt_logging_enabled(self.participant_identity)):
                    
                    if speech_event.alternatives:
                        transcript_text = speech_event.alternatives[0].text.strip()
                        if transcript_text:
                            # Log the transcription
                            logger.info(
                                f"[STT] Participant: {self.participant_identity}, "
                                f"Agent: {self.agent_identity}, "
                                f"Type: {speech_event.type}, "
                                f"Transcript: \"{transcript_text}\""
                            )
                
                yield speech_event
        
        return log_and_forward_speech()

