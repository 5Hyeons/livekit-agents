"""
ChatModeAgent class for text-only chat interaction.

This agent uses a regular LLM (Gemini 2.5 Flash) for cost-effective text chat.
It INCLUDES show_event_details tool as chat mode displays markdown in UI.
"""

import logging
import json

from config import create_cafe_show_instructions
from .model_factory import get_llm

from livekit.agents.llm import function_tool
from livekit.agents.voice.agent import Agent
from livekit.agents import RunContext, get_job_context

logger = logging.getLogger("chat-mode-agent")


class ChatModeAgent(Agent):
    """
    Chat Mode Agent with text-only interaction.

    This agent extends the base Agent class to provide:
    - Text-only chat via regular LLM (Gemini 2.5 Flash)
    - Cost-effective alternative to realtime models
    - show_event_details tool for displaying markdown UI
    - NO face animation (text mode only)
    """

    def __init__(
        self,
        user_data: dict,
        setup_data: dict,
        api_manager,  # RestAPIManager instance
        chat_ctx=None,  # Optional - for agent handoff
    ):
        """
        Initialize ChatModeAgent.

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
        logger.info(f"ChatModeAgent initialized for docent: {setup_data['docentId']}")

        # Initialize parent Agent with regular LLM (NOT realtime)
        super().__init__(
            instructions=base_instructions,
            llm=get_llm("chat"),  # Gemini 2.5 Flash
            chat_ctx=chat_ctx  # Auto-copied by Agent constructor
            # NO stf parameter - text only, no face animation
        )

    async def on_enter(self):
        """
        Handle agent entry into conversation session.
        """
        logger.info(f"Chat Mode Agent entering session for user: {self.userdata['user_id']}")
        if self.setup_data['docentId'] == 'None':
            system_context = "[SYSTEM_CONTEXT: First-time visitor. Welcome them briefly as CafeShow 2025 official AI (1-2 sentences).]"
        else:
            system_context = "[SYSTEM_CONTEXT: First-time visitor. Give a brief introduction (1-2 sentences) mentioning your company name and booth number.]"
        self.session.generate_reply(user_input=system_context)

    @function_tool()
    async def show_event_details(
        self,
        context: RunContext,
        topic: str,
    ) -> str:
        """Show detailed event information in the chat interface.

        WHEN TO USE THIS TOOL:
        - User asks about FORUM, CONFERENCE, SEMINAR
        - User asks about TICKETS, PRICING, BOOKING, REFUND
        - User asks about HALL layout, EXHIBITION structure
        - User asks about TRANSPORTATION, PARKING, SUBWAY
        - User asks about PROGRAMS, SCHEDULE

        USER INPUT EXAMPLES:
        USER: 입장 절차 알려줘.
        USER: 티켓 예매는 어떻게 해?
        USER: 오늘 프로그램 알려줘.
        USER: 오늘 스케줄 알려줘.
        USER: 지금 행사 하는거 있어?
        USER: 주차는 어떻게 하는거야?
        USER: 주요 프로그램
        USER: 티켓 가격
        USER: 주차

        Args:
            topic: MUST be one of: 'forum', 'ticket', 'hall', 'transportation', 'program'

        Returns:
            None or a string to indicate the result of the tool call
        """
        # Check if user is in chat mode (can see MD details)
        current_mode = context.userdata.get('current_mode', 'chat')

        if current_mode != 'chat':
            logger.info(f"[ChatModeAgent] Skipping tool - user in {current_mode} mode")
            return "[SYSTEM_CONTEXT: Provide detailed, helpful guidance to the user in a friendly conversational manner.]"

        try:
            # Access room via get_job_context() (official pattern from LiveKit docs)
            room = get_job_context().room
            # Get first remote participant (user)
            participant_identity = next(iter(room.remote_participants))

            # Send RPC to React frontend (ChatView will handle display)
            await room.local_participant.perform_rpc(
                destination_identity=participant_identity,
                method="show_event_details",
                payload=json.dumps({"topic": topic}),
                response_timeout=2.0,
            )

            logger.info(f"[ChatModeAgent] Sent detail view RPC for topic: {topic}")
            return None  # Silent completion (UI already updated)

        except Exception as e:
            logger.error(f"[ChatModeAgent] Failed to send detail RPC: {e}")
            return None  # Silent failure (don't confuse LLM)
