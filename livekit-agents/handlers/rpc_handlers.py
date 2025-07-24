"""
RPC (Remote Procedure Call) handlers for client-agent communication.
"""

import logging
from typing import TYPE_CHECKING

from livekit import rtc
from livekit.agents.voice.agent_session import AgentSession

if TYPE_CHECKING:
    pass

logger = logging.getLogger("rpc-handlers")


class RPCHandlers:
    """
    Manages RPC method handlers for client-agent communication.

    This class provides handlers for:
    - Agent interruption
    - Attention checking for inactive users
    """

    def __init__(self, session: AgentSession, user_language: str):
        """
        Initialize RPC handlers.

        Args:
            session: AgentSession instance for communication
            user_language: User's preferred language for messages
        """
        self.session = session
        self.user_language = user_language

    def create_interrupt_handler(self):
        """
        Create agent interruption RPC handler.

        Returns:
            Async function to handle agent interruption requests
        """

        async def interrupt_agent(data: rtc.RpcInvocationData) -> None:
            """Handle client request to interrupt agent."""
            logger.info(f"RPC 'interrupt_agent' called by: {data.caller_identity}")

            try:
                # Interrupt current agent activity
                await self.session.interrupt()
                logger.info("AgentSession interrupt completed")
            except Exception as e:
                logger.error(f"Error during agent interruption: {e}")

        return interrupt_agent

    def create_attention_check_handler(self):
        """
        Create attention check RPC handler for inactive users.

        Returns:
            Async function to handle attention check requests
        """

        async def check_attention(data: rtc.RpcInvocationData) -> None:
            """Handle attention check for users inactive for 1+ hours."""
            logger.info(f"RPC 'check_attention' called by: {data.caller_identity}")

            try:
                system_context = "[SYSTEM_CONTEXT: User inactive for over an hour. Check if they're still there naturally.]"
                await self.session.generate_reply(user_input=system_context)
                logger.info("Attention check generated")
            except Exception as e:
                logger.error(f"Error generating attention check: {e}")

        return check_attention
    
    def create_send_text_input_handler(self):
        """
        Create text input RPC handler for direct text messages from client.
        
        Returns:
            Async function to handle text input requests
        """
        
        async def send_text_input(data: rtc.RpcInvocationData) -> str:
            """Handle direct text input from Unity client."""
            logger.info(f"RPC 'send_text_input' called by: {data.caller_identity}")
            logger.info(f"Text input payload: {data.payload}")
            
            try:
                import json
                
                # Parse the payload
                payload = json.loads(data.payload)
                text = payload.get("text", "")
                
                if not text:
                    logger.warning("Empty text received in send_text_input RPC")
                    return json.dumps({"status": "error", "message": "Empty text"})
                
                # Generate reply using the text input
                logger.info(f"Processing text input: {text}")
                await self.session.generate_reply(user_input=text)
                
                return json.dumps({"status": "success"})
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON payload: {e}")
                return json.dumps({"status": "error", "message": "Invalid JSON"})
            except Exception as e:
                logger.error(f"Error processing text input: {e}")
                return json.dumps({"status": "error", "message": str(e)})
        
        return send_text_input

    def register_all_methods(self, local_participant: rtc.LocalParticipant):
        """
        Register all RPC methods with the local participant.

        Args:
            local_participant: LocalParticipant to register methods with
        """
        # Register RPC methods
        local_participant.register_rpc_method("interrupt_agent", self.create_interrupt_handler())
        local_participant.register_rpc_method(
            "check_attention", self.create_attention_check_handler()
        )
        local_participant.register_rpc_method(
            "send_text_input", self.create_send_text_input_handler()
        )

        logger.info("RPC methods registered: interrupt_agent, check_attention, send_text_input")
