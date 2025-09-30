"""
Main entry point for Wallmate Agent.

This module provides the main entry point and orchestrates all components:
- Session setup and configuration
- Agent initialization
- Event handler registration  
- RPC method registration
"""

import logging
import os

# Set higher logging level for Numba before other configurations
logging.getLogger('numba').setLevel(logging.WARNING)

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,  # Temporarily disabled
)
from livekit.agents.voice.agent_session import AgentSession
from livekit.plugins import silero

from core.wallmate_agent import WallmateAgent
from core.session_manager import setup_session
from core.rest_api_manager import RestAPIManager
from handlers.event_handlers import SessionEventHandlers
from handlers.rpc_handlers import RPCHandlers

# Load environment variables
load_dotenv()
logger = logging.getLogger("wallmate-main")

# Environment variables
DB_SERVER_URL = os.getenv("DB_SERVER_URL", "http://localhost:8018")



def prewarm(proc: JobProcess):
    """
    Prewarm function to initialize shared resources.
    
    Args:
        proc: Job process instance
    """
    # Load VAD model with optimized threshold
    proc.userdata["vad"] = silero.VAD.load(activation_threshold=0.4)
    logger.info("VAD model prewarmed successfully")


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for face animation agent.
    
    This function orchestrates the entire agent lifecycle:
    1. Connection and participant setup
    2. Session configuration and initialization
    3. Agent creation and startup
    4. Event handler registration
    5. RPC method registration
    
    Args:
        ctx: Job context from LiveKit framework
    """
    logger.info(f"Connecting to room: {ctx.room.name}")

    # Connect with audio-only subscription for STT
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for first participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting wallmate agent for participant: {participant.identity}")

    # Initialize RestAPIManager (MongoDB 직접 접근 제거!)
    api_manager = RestAPIManager(base_url=DB_SERVER_URL)

    # Setup session configuration
    setup_data = setup_session(participant)

    # Extract room input and output options
    room_input_options = setup_data['room_input_options']
    room_output_options = setup_data['room_output_options']

    # Get identity information
    user_identity = participant.identity
    thread_identity = f"{setup_data['scene_name']}_user-{user_identity}"

    # 1. Load user profile (장기 메모리) - 없으면 자동 생성
    user_profile = await api_manager.get_user_profile(thread_identity, user_id=user_identity, auto_create=True)

    # 2. Load conversation history (단기 메모리)
    chat_ctx = await api_manager.get_conversation(thread_identity, limit=50)

    # 3. Load token balance
    token_balance = await api_manager.get_token_balance(user_identity)

    # 4. Construct userdata
    user_profile = {
        "user_id": user_identity,
        "thread_id": thread_identity,
        "scene_id": setup_data["scene_name"],
        "name": user_profile.get("name", "Unknown"),
        "token_info": {
            **token_balance,
            "token_to_deduct": 0  # Session usage accumulator
        }
    }

    # Create agent instance
    agent = WallmateAgent(
        user_data=user_profile,
        setup_data=setup_data,
        chat_ctx=chat_ctx,  # 미리 로드한 대화 히스토리 주입!
        api_manager=api_manager  # RestAPIManager 전달
    )

    # Create agent session
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
        userdata=user_profile
    )

    # Create usage collector for metrics
    usage_collector = metrics.UsageCollector()

    # Create event handlers
    event_handlers = SessionEventHandlers(
        ctx=ctx,
        session=session,
        agent=agent,
        participant=participant,
        usage_collector=usage_collector,
        api_manager=api_manager  # RestAPIManager 전달
    )
    session.on("agent_state_changed", event_handlers.create_agent_state_handler())
    session.on("user_state_changed", event_handlers.create_user_state_handler())
    session.on("metrics_collected", event_handlers.create_metrics_handler())
    session.on("close", event_handlers.create_session_close_handler())
    
    # Start agent session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=room_input_options,
        room_output_options=room_output_options
    )
    
    # Create and register RPC handlers (MongoDB version - simplified)
    rpc_handlers = RPCHandlers(session)
    rpc_handlers.register_all_methods(ctx.room.local_participant)
    
    logger.info("Wallmate agent started successfully")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            shutdown_process_timeout= 10.0,
            drain_timeout=15.0,
            port=8085
        ),
    )