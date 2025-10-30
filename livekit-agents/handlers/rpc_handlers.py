"""
RPC (Remote Procedure Call) handlers for client-agent communication.
"""

import logging
import json
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
    - Text input from clients
    - Chat history management
    """

    def __init__(self, session: AgentSession):
        """
        Initialize RPC handlers.

        Args:
            session: AgentSession instance for communication
            db: UserDatabase instance for data operations
        """
        self.session = session

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

    def create_mode_change_handler(self):
        """
        Create user mode change RPC handler.

        Handles notification when user switches between chat and avatar modes.
        Updates session userdata and optionally interrupts agent.

        Returns:
            Async function to handle mode change requests
        """

        async def user_mode_changed(data: rtc.RpcInvocationData) -> None:
            """Handle user mode change (chat/avatar) notification."""
            logger.info(f"RPC 'user_mode_changed' called by: {data.caller_identity}")

            try:
                payload = json.loads(data.payload)
                mode = payload.get('mode', 'chat')
                should_interrupt = payload.get('should_interrupt', False)

                # Update session userdata with current mode
                self.session.userdata['current_mode'] = mode
                logger.info(f"User mode changed to: {mode}")

                # Interrupt agent if requested
                if should_interrupt:
                    await self.session.interrupt()
                    logger.info("Agent interrupted during mode change")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse mode change payload: {e}")
            except Exception as e:
                logger.error(f"Error handling mode change: {e}")

        return user_mode_changed

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
                self.session._update_user_state("listening")  # Update user state to listening
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
    
    # def create_clear_chat_history_handler(self):
    #     """
    #     Create chat history clearing RPC handler.
        
    #     Returns:
    #         Async function to handle chat history clearing requests
    #     """
        
    #     async def clear_chat_history(data: rtc.RpcInvocationData) -> str:
    #         """Handle client request to clear chat history."""
    #         logger.info(f"RPC 'clear_chat_history' called by: {data.caller_identity}")
            
    #         try:
    #             import json
                
    #             # Clear chat history from database
    #             deleted_count = self.db.clear_chat_history()
                
    #             # Clear current session's chat context while preserving system message
    #             try:
    #                 agent = self.session.current_agent
    #                 current_ctx = agent.chat_ctx
                    
    #                 # Find system message
    #                 system_message = None
    #                 for item in current_ctx.items:
    #                     if (item.type == "message" and 
    #                         hasattr(item, 'role') and 
    #                         item.role == "system"):
    #                         system_message = item
    #                         break
                    
    #                 # Create new context with only system message
    #                 from livekit.agents.llm import ChatContext
    #                 new_ctx = ChatContext.empty()
    #                 if system_message:
    #                     new_ctx.items.append(system_message)
                    
    #                 # Update agent's chat context
    #                 await agent.update_chat_ctx(new_ctx)
                    
    #                 logger.info("Successfully reset session chat context")
    #                 context_message = " 현재 세션의 대화 컨텍스트도 초기화되었습니다."
                    
    #             except Exception as ctx_error:
    #                 logger.warning(f"Failed to reset session chat context: {ctx_error}")
    #                 context_message = " (현재 세션의 컨텍스트 초기화는 실패했습니다)"
                
    #             return json.dumps({
    #                 "status": "success", 
    #                 "deleted_count": deleted_count,
    #                 "message": f"{deleted_count}개의 채팅 기록이 삭제되었습니다{context_message}"
    #             })
                
    #         except Exception as e:
    #             logger.error(f"Error clearing chat history: {e}")
    #             import json
    #             return json.dumps({"status": "error", "message": str(e)})
        
    #     return clear_chat_history

    def register_all_methods(self, local_participant: rtc.LocalParticipant):
        """
        Register all RPC methods with the local participant.

        Args:
            local_participant: LocalParticipant to register methods with
        """
        # Register RPC methods
        local_participant.register_rpc_method("interrupt_agent", self.create_interrupt_handler())
        local_participant.register_rpc_method("user_mode_changed", self.create_mode_change_handler())
        local_participant.register_rpc_method(
            "check_attention", self.create_attention_check_handler()
        )
        local_participant.register_rpc_method(
            "send_text_input", self.create_send_text_input_handler()
        )
        # local_participant.register_rpc_method(
        #     "clear_chat_history", self.create_clear_chat_history_handler()
        # )

        logger.info("RPC methods registered: interrupt_agent, check_attention, send_text_input, clear_chat_history")
