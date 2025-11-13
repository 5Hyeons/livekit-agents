"""
RPC (Remote Procedure Call) handlers for client-agent communication.
"""

import logging
import json
from typing import TYPE_CHECKING

from livekit import rtc
from livekit.agents.voice.agent_session import AgentSession

# Import agent classes for handoff
from core.avatar_mode_agent import AvatarModeAgent
from core.chat_mode_agent import ChatModeAgent

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
        Create user mode change RPC handler with agent handoff.

        Handles notification when user switches between chat and avatar modes.
        Performs agent handoff to switch between:
        - AvatarModeAgent (realtime voice) for avatar mode
        - ChatModeAgent (text-only) for chat mode

        Returns:
            Async function to handle mode change requests
        """

        async def user_mode_changed(data: rtc.RpcInvocationData) -> None:
            """Handle user mode change with agent handoff."""
            logger.info(f"RPC 'user_mode_changed' called by: {data.caller_identity}")

            try:
                payload = json.loads(data.payload)
                mode = payload.get('mode', 'chat')
                should_interrupt = payload.get('should_interrupt', False)

                # Get current agent and chat context
                current_agent = self.session.current_agent
                current_chat_ctx = current_agent.chat_ctx  # Read-only view

                # Get required data from userdata for agent recreation
                user_data = self.session.userdata.get('user_data')
                setup_data = self.session.userdata.get('setup_data')
                api_manager = self.session.userdata.get('api_manager')

                # Create appropriate agent based on mode
                if mode == 'avatar':
                    logger.info("Switching to AvatarModeAgent (realtime voice)")
                    new_agent = AvatarModeAgent(
                        user_data=user_data,
                        setup_data=setup_data,
                        api_manager=api_manager,
                        chat_ctx=current_chat_ctx  # Auto-copied by Agent constructor
                    )
                    # Enable audio and animation output for avatar mode
                    self.session.output.set_audio_enabled(True)
                    self.session.output.set_animation_enabled(True)
                    logger.info("Audio and animation output enabled")

                else:  # 'chat' mode
                    logger.info("Switching to ChatModeAgent (text-only)")
                    new_agent = ChatModeAgent(
                        user_data=user_data,
                        setup_data=setup_data,
                        api_manager=api_manager,
                        chat_ctx=current_chat_ctx  # Auto-copied by Agent constructor
                    )
                    # Disable audio and animation output for chat mode (text-only)
                    self.session.output.set_audio_enabled(False)
                    self.session.output.set_animation_enabled(False)
                    logger.info("Audio and animation output disabled (text-only mode)")

                # Update session userdata
                self.session.userdata['current_mode'] = mode

                # Perform agent handoff
                self.session.update_agent(new_agent)
                logger.info(f"Agent handoff complete to {mode} mode")

                # Interrupt agent if requested
                if should_interrupt:
                    await self.session.interrupt()
                    logger.info("Agent interrupted during mode change")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse mode change payload: {e}")
            except Exception as e:
                logger.error(f"Error during agent handoff: {e}", exc_info=True)

        return user_mode_changed

    def create_language_change_handler(self):
        """
        Create language change RPC handler.

        Handles notification when user switches language.
        Updates agent instructions with new language and clears chat history
        to provide a fresh start in the new language.

        Returns:
            Async function to handle language change requests
        """

        async def user_language_changed(data: rtc.RpcInvocationData) -> str:
            """Handle user language change with instruction update and history clear."""
            logger.info(f"RPC 'user_language_changed' called by: {data.caller_identity}")

            try:
                payload = json.loads(data.payload)
                new_language = payload.get('language', 'ko')  # ISO language code

                # Get current agent
                current_agent = self.session.current_agent
                current_chat_ctx = current_agent.chat_ctx

                # Get required data from userdata
                setup_data = self.session.userdata.get('setup_data')
                docent_id = setup_data.get('docentId', '')

                # Update setup_data with new language
                setup_data['language'] = new_language
                logger.info(f"Language changed to: {new_language}")

                # Recreate instructions with new language
                from config.personas import create_cafe_show_instructions
                new_instructions = create_cafe_show_instructions(new_language, docent_id)

                # Clear chat history while preserving system message
                system_message = None
                for item in current_chat_ctx.items:
                    if (item.type == "message" and
                        hasattr(item, 'role') and
                        item.role in ["system", "developer"]):
                        system_message = item
                        break

                # Create new context with only system message
                from livekit.agents.llm import ChatContext
                new_chat_ctx = ChatContext.empty()
                if system_message:
                    new_chat_ctx.items.append(system_message)
                logger.info("Chat history cleared (system message preserved)")

                # Update agent instructions and chat context
                await current_agent.update_instructions(new_instructions)
                await current_agent.update_chat_ctx(new_chat_ctx)
                logger.info("Agent instructions and chat context updated")

                # Always interrupt agent during language change
                await self.session.interrupt()
                logger.info("Agent interrupted for language change")

                return json.dumps({"status": "success", "language": new_language})

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse language change payload: {e}")
                return json.dumps({"status": "error", "message": "Invalid JSON payload"})
            except Exception as e:
                logger.error(f"Error during language change: {e}", exc_info=True)
                return json.dumps({"status": "error", "message": str(e)})

        return user_language_changed

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
        local_participant.register_rpc_method("user_language_changed", self.create_language_change_handler())
        local_participant.register_rpc_method(
            "check_attention", self.create_attention_check_handler()
        )
        local_participant.register_rpc_method(
            "send_text_input", self.create_send_text_input_handler()
        )
        # local_participant.register_rpc_method(
        #     "clear_chat_history", self.create_clear_chat_history_handler()
        # )

        logger.info("RPC methods registered: interrupt_agent, user_mode_changed, user_language_changed, check_attention, send_text_input")
