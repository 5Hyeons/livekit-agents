"""
Main entry point for Cafe Show Agent.

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

from core.avatar_mode_agent import AvatarModeAgent
from core.session_manager import setup_session
from core.rest_api_manager import RestAPIManager
from handlers.event_handlers import SessionEventHandlers
from handlers.rpc_handlers import RPCHandlers

# Load environment variables
load_dotenv()
logger = logging.getLogger("cafe-show-main")

# Environment variables
DB_SERVER_URL = os.getenv("DB_SERVER_URL", "http://localhost:8028")



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
    logger.info(f"Starting cafe show agent for participant: {participant.identity}")

    # Initialize RestAPIManager (MongoDB 직접 접근 제거!)
    api_manager = RestAPIManager(base_url=DB_SERVER_URL)

    # Setup session configuration
    setup_data = setup_session(participant)

    # Extract room input and output options
    room_input_options = setup_data['room_input_options']
    room_output_options = setup_data['room_output_options']

    # Get identity information
    user_identity = participant.identity

    # Construct user profile
    user_profile = {
        "user_id": user_identity,
        "current_mode": "avatar",  # Default mode: avatar (voice)
    }

    # Construct session userdata with data needed for agent handoff
    session_userdata = {
        "user_id": user_identity,
        "current_mode": "avatar",  # Default mode: avatar (voice)
        # Required for agent recreation during handoff
        "user_data": user_profile,
        "setup_data": setup_data,
        "api_manager": api_manager
    }

    # Create initial agent (AvatarModeAgent by default)
    agent = AvatarModeAgent(
        user_data=user_profile,
        setup_data=setup_data,
        api_manager=api_manager  # RestAPIManager 전달
        # chat_ctx=None (fresh start)
    )

    # Create agent session
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
        userdata=session_userdata  # Store handoff data here
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
    
    logger.info("Cafe show agent started successfully")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            shutdown_process_timeout= 10.0,
            drain_timeout=15.0,
            port=8096
        ),
    )