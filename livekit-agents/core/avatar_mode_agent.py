"""
AvatarModeAgent class for realtime voice interaction with face animation.

This agent uses OpenAI Realtime API for low-latency voice interaction.
It does NOT include show_event_details tool as avatar mode doesn't display markdown.
"""

import logging

from config import create_cafe_show_instructions
from .model_factory import get_stf
from livekit.plugins import openai
from livekit.agents.voice.agent import Agent

logger = logging.getLogger("avatar-mode-agent")


class AvatarModeAgent(Agent):
    """
    Avatar Mode Agent with realtime voice and face animation.

    This agent extends the base Agent class to provide:
    - Realtime voice interaction via OpenAI Realtime API
    - Face animation capabilities via STF (Speech-To-Face)
    - Multilingual conversation support
    - NO show_event_details tool (avatar mode doesn't show markdown UI)
    """

    def __init__(
        self,
        user_data: dict,
        setup_data: dict,
        api_manager,  # RestAPIManager instance
        chat_ctx=None,  # Optional - for agent handoff
    ):
        """
        Initialize AvatarModeAgent.

        Args:
            user_data: User data including user_id
            setup_data: Setup data containing language, docentId
            api_manager: RestAPIManager for database operations via REST API
            chat_ctx: Optional ChatContext for preserving conversation history during handoff
        """
        self.userdata = user_data
        self.api_manager = api_manager
        self.setup_data = setup_data  # Store for access

        # Create base instructions with persona
        base_instructions = create_cafe_show_instructions(
            setup_data['language'],
            setup_data['docentId']
        )
        logger.info(f"AvatarModeAgent initialized for docent: {setup_data['docentId']}")

        # Select voice based on docent mode
        if setup_data['docentId'] != 'None':
            # Docent mode - male voice
            voice = "cedar"
            logger.info(f"Docent mode: Using male voice 'cedar'")
        else:
            # General AI mode - female voice
            voice = "sage"
            logger.info(f"General AI mode: Using female voice 'ash'")

        # Create realtime model with selected voice
        llm = openai.realtime.RealtimeModel(
            model="gpt-realtime",
            voice=voice
        )

        # Initialize parent Agent with realtime model
        super().__init__(
            instructions=base_instructions,
            llm=llm,  # OpenAI Realtime API with selected voice
            stf=get_stf(),  # Face animation
            chat_ctx=chat_ctx  # Auto-copied by Agent constructor
        )

    async def on_enter(self):
        """
        Handle agent entry into conversation session.
        """
        logger.info(f"Avatar Mode Agent entering session for user: {self.userdata['user_id']}")
        if self.setup_data['docentId'] == 'None':
            system_context = "[SYSTEM_CONTEXT: First-time visitor. Welcome them briefly as CafeShow 2025 official AI (1-2 sentences).]"
        else:
            system_context = "[SYSTEM_CONTEXT: First-time visitor. Give a brief introduction (1-2 sentences) mentioning your company name and booth number.]"
        self.session.generate_reply(user_input=system_context)

    # NO show_event_details tool here!
    # Avatar mode doesn't display markdown details in UI
