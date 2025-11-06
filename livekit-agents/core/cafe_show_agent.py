"""
CafeShowAgent class for LiveKit voice AI agents with face animation support.
"""

import logging
import json

from config import (
    create_cafe_show_instructions,
)
from .model_factory import get_stt, get_tts, get_stf, get_llm

from livekit.agents.llm import ChatContext, FunctionTool, function_tool
from livekit.agents.voice.agent import Agent, ModelSettings
from livekit.agents import RunContext, get_job_context
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
        api_manager,  # RestAPIManager instance
    ):
        """
        Initialize CafeShowAgent.

        Args:
            user_data: User data including user_id, thread_id, name, credit_info
            setup_data: Setup data containing language, persona, voice, model, scene
            api_manager: RestAPIManager for database operations via REST API
        """
        self.userdata = user_data
        self.api_manager = api_manager
        self.setup_data = setup_data  # Store for tool access

        # Create base instructions with persona
        base_instructions = create_cafe_show_instructions(
            setup_data['language'],
            setup_data['docentId']
        )
        logger.info(f"Docent ID: {setup_data['docentId']}")

        # Initialize parent Agent with pre-loaded chat context
        super().__init__(
            instructions=base_instructions,
            llm=get_llm("realtime"),
            stf=get_stf(),
        )

    async def on_enter(self):
        """
        Handle agent entry into conversation session.
        """
        logger.info(f"Cafe Show Agent entering session for user: {self.userdata['user_id']}")
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
            logger.info(f"[CafeShowAgent] Skipping tool - user in {current_mode} mode")
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

            logger.info(f"[CafeShowAgent] Sent detail view RPC for topic: {topic}")
            return None  # Silent completion (UI already updated)

        except Exception as e:
            logger.error(f"[CafeShowAgent] Failed to send detail RPC: {e}")
            return None  # Silent failure (don't confuse LLM)

    # @function_tool()
    # async def search_similar_companies(
    #     self,
    #     context: RunContext,
    #     query: str,
    #     max_results: int = 3
    # ) -> str:
    #     """Search for companies similar to a query or within a specific category.

    #     WHEN TO USE THIS TOOL:
    #     - User asks "비슷한 회사 있어?" (similar companies?)
    #     - User asks about product categories (e.g., "커피머신 파는 곳", "포장재 업체")
    #     - User wants booth recommendations based on interests
    #     - User asks for comparisons between companies

    #     USER INPUT EXAMPLES:
    #     USER: 비슷한 회사 또 있어?
    #     USER: 커피머신 파는 곳 알려줘
    #     USER: 포장재 업체 추천해줘
    #     USER: 초콜릿 관련 부스는?

    #     Args:
    #         query: Search terms (e.g., "친환경 포장", "에스프레소 머신", "초콜릿")
    #         max_results: Number of results to return (default 3, max 5)

    #     Returns:
    #         Formatted list of matching companies with booth numbers and introductions
    #     """
    #     from pathlib import Path

    #     try:
    #         # Load docents and categories
    #         config_dir = Path(__file__).parent.parent / "config"
    #         docents_file = config_dir / "docents.json"
    #         categories_file = config_dir / "docent_categories.json"

    #         with open(docents_file, 'r', encoding='utf-8') as f:
    #             all_docents = json.load(f)

    #         # Try to load categories (may not exist yet)
    #         categories_data = {}
    #         try:
    #             with open(categories_file, 'r', encoding='utf-8') as f:
    #                 categories_data = json.load(f)
    #         except FileNotFoundError:
    #             logger.warning("[search_similar_companies] Categories file not found, using keyword search only")

    #         matches = []
    #         query_lower = query.lower()
    #         current_docent_id = self.setup_data.get('docentId')

    #         # Strategy 1: Try category-based search first
    #         if categories_data and 'docent_to_categories' in categories_data:
    #             docent_to_cats = categories_data.get('docent_to_categories', {})
    #             categories = categories_data.get('categories', {})

    #             # Find matching categories
    #             matching_category_names = []
    #             for cat_name in categories.keys():
    #                 if query_lower in cat_name.lower():
    #                     matching_category_names.append(cat_name)

    #             # Get docents from matching categories
    #             for cat_name in matching_category_names:
    #                 docent_ids = categories.get(cat_name, [])
    #                 for docent_id in docent_ids:
    #                     if docent_id == current_docent_id:
    #                         continue  # Skip current booth
    #                     if docent_id in all_docents:
    #                         matches.append({
    #                             'id': docent_id,
    #                             'booth': all_docents[docent_id].get('boothNumber'),
    #                             'name': all_docents[docent_id].get('koreanCompanyName'),
    #                             'intro': all_docents[docent_id].get('shortIntro', '')[:150]
    #                         })

    #         # Strategy 2: Keyword-based search in descriptions
    #         if len(matches) < max_results:
    #             for docent_id, docent_data in all_docents.items():
    #                 if docent_id == current_docent_id:
    #                     continue

    #                 if any(m['id'] == docent_id for m in matches):
    #                     continue  # Already added

    #                 # Search in text fields
    #                 searchable_text = (
    #                     docent_data.get('shortIntro', '') + ' ' +
    #                     docent_data.get('descriptionKo', '') + ' ' +
    #                     docent_data.get('descriptionEn', '') + ' ' +
    #                     docent_data.get('koreanCompanyName', '')
    #                 ).lower()

    #                 if query_lower in searchable_text:
    #                     matches.append({
    #                         'id': docent_id,
    #                         'booth': docent_data.get('boothNumber'),
    #                         'name': docent_data.get('koreanCompanyName'),
    #                         'intro': docent_data.get('shortIntro', '')[:150]
    #                     })

    #                 if len(matches) >= max_results * 2:  # Get extras to filter best
    #                     break

    #         # Limit results
    #         matches = matches[:min(max_results, 5)]

    #         # Format response
    #         if matches:
    #             result = f"[SYSTEM_CONTEXT: Found {len(matches)} similar companies. Provide enthusiastic, helpful recommendations in a bright tone.]\n\n"
    #             for m in matches:
    #                 result += f"**부스 {m['booth']}** - {m['name']}\n{m['intro']}\n\n"
    #             return result
    #         else:
    #             return "[SYSTEM_CONTEXT: No exact matches found. Suggest browsing different hall areas or checking the event directory. Maintain cheerful tone.]"

    #     except Exception as e:
    #         logger.error(f"[search_similar_companies] Error: {e}")
    #         return "[SYSTEM_CONTEXT: Search temporarily unavailable. Suggest visiting the information desk or browsing hall areas. Stay positive and helpful.]"
