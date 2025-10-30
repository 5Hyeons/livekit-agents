"""
CafeShowAgent class for LiveKit voice AI agents with face animation support.
"""

import logging

from config import (
    create_cafe_show_instructions,
)
from .model_factory import get_stt, get_tts, get_stf, get_llm

from livekit.agents.llm import ChatContext, FunctionTool, function_tool
from livekit.agents.voice.agent import Agent, ModelSettings
from livekit.rtc import AudioFrame

logger = logging.getLogger("cafe-show-agent")



class CafeShowAgent(Agent):
    """
    Cafe Show Agent with multilingual support and performance tracking.

    This agent extends the base Agent class to provide:
    - Face animation capabilities via STF (Speech-To-Face)
    - Multilingual conversation support
    - Performance metrics tracking
    - User data persistence
    - Custom persona support
    """

    def __init__(
        self,
        user_data: dict,
        setup_data: dict,
        chat_ctx: ChatContext,  # Pre-loaded conversation history
        api_manager,  # RestAPIManager instance
    ):
        """
        Initialize CafeShowAgent.

        Args:
            user_data: User data including user_id, thread_id, name, credit_info
            setup_data: Setup data containing language, persona, voice, model, scene
            chat_ctx: Pre-loaded ChatContext with conversation history
            api_manager: RestAPIManager for database operations via REST API
        """
        self.userdata = user_data
        self.api_manager = api_manager

        if setup_data['custom_persona']:
            logger.info(f"Custom persona: {setup_data['custom_persona']}")

        # Create base instructions with persona
        base_instructions = create_cafe_show_instructions(
            setup_data['agent_language'], custom_persona=setup_data['custom_persona']
        )

        logger.info(f"Using REST API for memory management (loaded {len(chat_ctx.items)} messages)")

        # Initialize parent Agent with pre-loaded chat context
        super().__init__(
            instructions=base_instructions,
            chat_ctx=chat_ctx,  # 미리 로드한 대화 히스토리
            llm=get_llm("realtime"),
            stf=get_stf(),
        )

    async def on_enter(self):
        """
        Handle agent entry into conversation session.
        Memory is already loaded in main.py and injected into chat_ctx.
        """
        logger.info(f"Cafe Show eAgent entering session for user: {self.userdata['user_id']}")
