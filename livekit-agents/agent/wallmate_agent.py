"""
WallmateAgent class for LiveKit voice AI agents with face animation support.
"""

import logging
import os
import wave
from collections.abc import AsyncIterable
from datetime import datetime

from config import (
    get_stt, get_llm, get_tts, get_stf, create_instructions,
    is_tts_logging_enabled, get_tts_output_directory, ensure_logging_directories
)
from user_database import UserData, UserDatabase

from livekit.agents import utils
from livekit.agents.llm import function_tool
from livekit.agents.voice.agent import Agent, ModelSettings
from livekit.rtc import AudioFrame

logger = logging.getLogger("wallmate-agent")


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
        user_data: UserData,
        db: UserDatabase,
        participant_identity: str,
        agent_identity: str,
        user_language: str = "ko",
        custom_persona: str = "",
        voice_name: str = "FEMALE_1",
        model_name: str = "claude-4-sonnet-20250514",
    ):
        """
        Initialize WallmateAgent with user context and configuration.

        Args:
            user_data: User data from database
            db: Database instance for persistence
            participant_identity: Identity of the participant
            agent_identity: Identity of the agent
            user_language: User's preferred language (ko, en, ja, zh)
            custom_persona: Custom personality instructions
            voice_name: Voice preset name (FEMALE_1/2, MALE_1/2)
            model_name: LLM model name (claude-4-sonnet-20250514, gpt-4o-mini, etc.)
        """
        self.user_data = user_data
        self.db = db
        self.participant_identity = participant_identity
        self.agent_identity = agent_identity
        self.user_language = user_language
        self.custom_persona = custom_persona
        self._preloaded_message_count = 120  # Number of messages to load from history
        
        # Initialize logging directories for this user
        self.logging_enabled = is_tts_logging_enabled(self.participant_identity)
        self.tts_output_dir = None
        
        if self.logging_enabled:
            logging_dirs = ensure_logging_directories(self.participant_identity)
            self.tts_output_dir = logging_dirs["tts_output"]
            logger.info(f"TTS logging enabled for user: {self.participant_identity}")
        else:
            logger.debug(f"TTS logging disabled for user: {self.participant_identity}")

        if self.custom_persona:
            logger.info(f"Use Custom persona: {self.custom_persona}")

        # Performance tracking is now handled in event_handlers.py

        # Create base instructions with persona
        base_instructions = create_instructions(
            self.user_language, custom_persona=self.custom_persona
        )

        # Load previous conversation history into chat context
        chat_ctx = self._prepare_chat_context_with_history(self._preloaded_message_count)
        self._preloaded_message_count = len(chat_ctx.items)
        
        logger.info(
            f"Loaded {self._preloaded_message_count} messages from conversation history"
        )

        # Initialize parent Agent with simple config
        super().__init__(
            instructions=base_instructions,
            chat_ctx=chat_ctx,
            stt=get_stt(self.user_language),
            llm=get_llm(model_name),
            tts=get_tts(voice_name),
            stf=get_stf(),
        )

    def _prepare_chat_context_with_history(self, message_count: int):
        """
        Prepare chat context with previous conversation history.

        Args:
            message_count: Number of messages to load from history

        Returns:
            ChatContext with loaded conversation history
        """
        # Import ChatContext from llm module
        from livekit.agents.llm import ChatContext

        # Create new chat context
        chat_ctx = ChatContext()

        # Get recent conversation context
        context = self.db.get_recent_context(
            self.user_data.participant_id, message_count=message_count
        )

        if not context:
            return chat_ctx

        # Parse context and add to chat_ctx
        lines = context.split("\n")
        for line in lines:
            if line.startswith("user: "):
                content = line[6:]  # Remove "user: " prefix
                if content.strip():
                    chat_ctx.add_message(role="user", content=content)
            elif line.startswith("assistant: "):
                content = line[11:]  # Remove "assistant: " prefix
                if content.strip():
                    chat_ctx.add_message(role="assistant", content=content)
            # Skip separator lines like "---"

        return chat_ctx
    

    async def on_enter(self):
        """
        Handle agent entry into conversation session.

        Generates appropriate greeting based on whether user is new or returning.
        """
        logger.info(f"WallmateAgent entering session for user: {self.user_data.participant_id}")

        if self.user_data.display_name:
            # Returning user - generate personalized greeting
            system_context = f"[SYSTEM_CONTEXT: User '{self.user_data.display_name}' just joined. You've met before. Greet naturally.]"
            await self.session.generate_reply(user_input=system_context, allow_interruptions=False)
            logger.info(f"Generated returning user greeting for: {self.user_data.display_name}")
        else:
            # New user - generate introduction greeting
            system_context = "[SYSTEM_CONTEXT: User just joined. This is first meeting. Introduce yourself naturally.]"
            await self.session.generate_reply(user_input=system_context, allow_interruptions=False)
            logger.info("Generated new user greeting")

    @function_tool
    async def save_user_name(self, name: str) -> str:
        """
        Save user's name when they introduce themselves.

        Args:
            name: The user's name

        Returns:
            Confirmation message in user's language
        """
        # Prevent duplicate saves
        if self.user_data.display_name and self.user_data.display_name == name:
            logger.info(f"Name already saved: {name}")
            return ""

        # Update database and local data
        self.db.update_user_name(self.user_data.participant_id, name)
        self.user_data.display_name = name
        logger.info(f"User name saved: {self.user_data.participant_id} -> {name}")

        # Return system context for the agent to acknowledge
        return f"[SYSTEM_CONTEXT: User introduced themselves as '{name}'. Acknowledge this naturally and continue the conversation.]"
    
    def _save_audio_frames_as_wav(self, audio_frames: list[AudioFrame], text_context: str = ""):
        """
        Save accumulated audio frames as a WAV file for allowed users only.
        
        Args:
            audio_frames: List of audio frames to save
            text_context: Text that was synthesized (for filename)
        """
        # Check if logging is enabled for this user
        if not self.logging_enabled or not self.tts_output_dir or not audio_frames:
            return
            
        try:
            # Generate unique filename with timestamp and identities
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds precision
            safe_participant = self.participant_identity.replace("/", "_").replace("\\", "_")
            safe_agent = self.agent_identity.replace("/", "_").replace("\\", "_")
            filename = f"{timestamp}_{safe_participant}_{safe_agent}.wav"
            filepath = os.path.join(self.tts_output_dir, filename)
            
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
                self._save_audio_frames_as_wav(collected_audio_frames, full_text)
        
        return collect_and_forward_audio()

