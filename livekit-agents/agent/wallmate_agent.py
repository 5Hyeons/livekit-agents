"""
WallmateAgent class for LiveKit voice AI agents with face animation support.
"""

import logging
import os
import sys
from typing import Any

from config.base_instructions import create_base_instructions
from config.voice_config import ElevenLabsConfig
from user_database import UserData, UserDatabase

from livekit.agents.llm import function_tool
from livekit.agents.stf import FaceAnimator, OutputMode
from livekit.agents.voice.agent import Agent
from livekit.plugins import deepgram, openai

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
        user_language: str = "ko",
        custom_persona: str = "",
        voice_name: str = "FEMALE_1",
    ):
        """
        Initialize WallmateAgent with user context and configuration.

        Args:
            user_data: User data from database
            db: Database instance for persistence
            user_language: User's preferred language (ko, en, ja, zh)
            custom_persona: Custom personality instructions
            voice_name: Voice preset name (FEMALE_1/2, MALE_1/2)
        """
        self.user_data = user_data
        self.db = db
        self.user_language = user_language
        self.custom_persona = custom_persona
        self._preloaded_message_count = 120  # Number of messages to load from history

        if self.custom_persona:
            logger.info(f"Use Custom persona: {self.custom_persona}")

        # Initialize reactivity tracker for performance monitoring
        sys.path.append(os.path.dirname(__file__))
        from streaming_reactivity_tracker import StreamingReactivityTracker

        self.reactivity_tracker = StreamingReactivityTracker()

        # Create base instructions with persona
        base_instructions = create_base_instructions(
            self.user_language, custom_persona=self.custom_persona
        )

        # Configure voice settings
        elevenlabs_config = ElevenLabsConfig.from_voice_name(voice_name)

        # Load previous conversation history into chat context
        chat_ctx = self._prepare_chat_context_with_history(self._preloaded_message_count)
        self._preloaded_message_count = len(chat_ctx.items)
        
        logger.info(
            f"Loaded {self._preloaded_message_count} messages from conversation history"
        )

        # Initialize parent Agent with components
        super().__init__(
            instructions=base_instructions,
            chat_ctx=chat_ctx,
            stt=deepgram.STT(model="nova-2-general", language=self.user_language),
            llm=openai.LLM(model="gpt-4o"),
            tts=elevenlabs_config.create_tts(),
            stf=FaceAnimator(chunk_duration_sec=0.5, output_mode=OutputMode.ANIMATION_WITH_AUDIO),
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

    def _update_metrics_data(self, metrics_obj):
        """
        Update reactivity metrics with streaming-optimized tracking.

        Args:
            metrics_obj: Metrics object from LiveKit agents framework
        """
        # Pass all metrics to streaming reactivity tracker
        # Optimized for streaming pipeline TTFT/TTFB measurement
        self.reactivity_tracker.record_metrics(metrics_obj)

    def _log_performance_summary(self):
        """Log performance summary at session end."""
        current_metrics = self.reactivity_tracker.get_current_metrics()
        if any(v is not None for v in current_metrics.values()):
            logger.info(f"Final reactivity metrics: {current_metrics}")

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get current performance statistics.

        Returns:
            Dictionary containing current reactivity metrics
        """
        return self.reactivity_tracker.get_current_metrics()
