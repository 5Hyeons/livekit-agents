"""
Main entry point for Wallmate Agent.

This module provides the main entry point and orchestrates all components:
- Session setup and configuration
- Agent initialization
- Event handler registration  
- RPC method registration
"""

import logging

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

from agent.wallmate_agent import WallmateAgent
from config import setup_session
from config.mongodb_config import get_store, test_mongodb_connection
from handlers.event_handlers import SessionEventHandlers  # Temporarily disabled
from handlers.rpc_handlers import RPCHandlers

# Load environment variables
load_dotenv()
logger = logging.getLogger("wallmate-main")


def get_or_create_user_profile(participant_id: str) -> dict:
    """
    Get existing user profile or create a new one.
    
    Args:
        participant_id: The participant's identity
    
    Returns:
        User profile data dictionary
    """
    if not participant_id:
        return {"name": "Unknown"}
    
    store = get_store()
    
    # Try to get existing profile
    profile_data = store.get(
        namespace=("user_profile", participant_id),
        key="basic_info"
    )
    
    if profile_data and profile_data.value:
        logger.info(f"[UserProfile] Loaded existing profile for: {participant_id}")
        return profile_data.value
    else:
        # Create new profile with default values
        from datetime import datetime
        new_profile = {
            "name": "Unknown",
            "created_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "token_info": {
                "total_granted": 10000,
                "total_used": 0,
                "remaining": 10000,
                "status": "normal"
            }
        }
        
        # Save to MongoDB Store
        store.put(
            namespace=("user_profile", participant_id),
            key="basic_info",
            value=new_profile
        )
        
        logger.info(f"[UserProfile] Created new profile for: {participant_id}")
        return new_profile
            


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
    
    # Setup session configuration
    setup_data = setup_session(participant)
    
    # Extract room input and output options
    room_input_options = setup_data['room_input_options']
    room_output_options = setup_data['room_output_options']
    
    # Get identity information
    participant_identity = participant.identity
    agent_identity = ctx.room.local_participant.identity
    
    # Get or create user profile
    user_profile = get_or_create_user_profile(participant_identity)
    
    # Create agent session
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
        userdata=user_profile
        )
    
    # Create usage collector for metrics (temporarily disabled)
    usage_collector = metrics.UsageCollector()
    
    # Create agent instance (MongoDB version)
    agent = WallmateAgent(
        participant_identity, 
        agent_identity,
        setup_data
    )
    
    # Create event handlers (MongoDB version)
    # Event handlers temporarily disabled - MongoDB Checkpointer handles core functionality
    event_handlers = SessionEventHandlers(
        ctx=ctx,
        session=session,
        agent=agent,
        participant=participant,
        usage_collector=usage_collector
    )
    session.on("agent_state_changed", event_handlers.create_agent_state_handler())
    session.on("user_state_changed", event_handlers.create_user_state_handler())
    session.on("metrics_collected", event_handlers.create_metrics_handler())
    session.on("session_close", event_handlers.create_session_close_handler())
    
    
    # Log agent identity
    agent_identity = ctx.room.local_participant.identity
    logger.info(f"Agent identity: {agent_identity}")
    
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