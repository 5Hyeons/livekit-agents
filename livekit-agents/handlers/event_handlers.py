"""
Event handlers for agent session management.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING

# User inactivity timeout configuration
USER_INACTIVITY_TIMEOUT_SECONDS = 180
CLOSE_SESSION_AFTER_INACTIVITY_SECONDS = 300
from user_database import ChatMessage, UserData, UserDatabase
from config import is_metrics_logging_enabled, get_metrics_output_directory, ensure_logging_directories

from livekit import rtc
from livekit.agents import JobContext, llm, metrics
from livekit.agents.voice import Agent, AgentSession, MetricsCollectedEvent, CloseEvent, CloseReason


logger = logging.getLogger("event-handlers")


class SessionEventHandlers:
    """
    Manages all session-level event handlers for face animation agent.

    This class encapsulates event handling logic for:
    - Agent state changes
    - User state changes
    - Session closure
    - Metrics collection
    """

    def __init__(
        self,
        ctx: JobContext,
        session: AgentSession,
        agent: Agent,
        participant: rtc.RemoteParticipant,
        db: UserDatabase,
        user_data: UserData,
        usage_collector: metrics.UsageCollector = None,
    ):
        """
        Initialize event handlers with required dependencies.

        Args:
            agent: Agent instance
            participant: Remote participant
            ctx: Job context
            db: Database instance
            user_data: User data
            usage_collector: Metrics collector
        """
        self.agent = agent
        self.participant = participant
        self.ctx = ctx
        self.db = db
        self.user_data = user_data
        self.usage_collector = usage_collector
        
        # Inactivity timeout tracking
        self.inactivity_task = None
        self.session = session
        
        # Session state tracking
        self._session_closing = False  # Prevent duplicate session close handling
        
        # E2E metrics tracking
        self.last_eou_time = None  # Track user speech end time
        
        self.eou_ms = None
        self.stt_ms = None
        self.llm_ttft = None  # LLM Time to First Token
        self.tts_ttfb = None  # TTS Time to First Byte
        self.stf_ttff = None  # STF Time to First Frame
        self.e2e_latency = None  # E2E Response Time
        self.metrics_logged = False  # Prevent duplicate logging
        
        # Initialize metrics logging for this user
        self.metrics_logging_enabled = is_metrics_logging_enabled(self.participant.identity)
        self.metrics_output_dir = None
        
        if self.metrics_logging_enabled:
            logging_dirs = ensure_logging_directories(self.participant.identity)
            self.metrics_output_dir = logging_dirs["metrics"]
            logger.info(f"Metrics logging enabled for user: {self.participant.identity}")
        else:
            logger.debug(f"Metrics logging disabled for user: {self.participant.identity}")
        
        # Token status tracking for RPC notifications
        self._last_token_status = "normal"  # Track previous status to detect changes
        self._token_thresholds = {
            "critical": 200,
            "low": 500
        }

    def create_agent_state_handler(self):
        """
        Create agent state change event handler.

        Returns:
            Event handler function for agent state changes
        """

        def on_agent_state_changed(ev):
            """Handle agent state change events - track speaking state."""
            if ev.new_state == "speaking":
                # Calculate E2E metrics directly when agent starts speaking
                event_timestamp = getattr(ev, 'created_at', None) or time.time()
                if self.last_eou_time is not None:
                    e2e_latency = event_timestamp - self.last_eou_time
                    if e2e_latency >= 0:  # Valid E2E measurement
                        self.e2e_latency = e2e_latency * 1000  # Store in ms
                        # Reset for next measurement
                        self.last_eou_time = None
                    else:
                        logger.warning(f"Invalid E2E latency: {e2e_latency * 1000:.1f}ms")

            logger.info(f"Agent state changed: {ev.old_state} -> {ev.new_state}")

            # Send agent state change notification to client via RPC
            try:
                payload = json.dumps(
                    {"old_state": ev.old_state, "new_state": ev.new_state, "timestamp": time.time()}
                )

                # Execute async RPC from sync handler
                task = asyncio.create_task(
                    self.ctx.room.local_participant.perform_rpc(
                        destination_identity=self.participant.identity,
                        method="agent_state_changed",
                        payload=payload,
                        response_timeout=1.0,
                    )
                )

                # Add completion callback for error logging
                def handle_rpc_result(future):
                    try:
                        future.result()
                        logger.debug(f"Agent state RPC sent successfully: {ev.new_state}")
                    except Exception as e:
                        logger.warning(f"Failed to send agent state RPC: {e}")

                task.add_done_callback(handle_rpc_result)

            except Exception as e:
                logger.error(f"Error sending agent state RPC: {e}")

        return on_agent_state_changed

    def create_user_state_handler(self):
        """
        Create user state change event handler.

        Returns:
            Event handler function for user state changes
        """

        async def inactive_user_timeout():
            """Handle user inactivity timeout after configured seconds."""
            if USER_INACTIVITY_TIMEOUT_SECONDS is None:
                return
            
            try:
                # Wait for the configured timeout period
                await asyncio.sleep(USER_INACTIVITY_TIMEOUT_SECONDS)
                
                await self.session.generate_reply(user_input="[SYSTEM_CONTEXT: User inactive for too long, So you are going to close the session. say goodbye to the user.]", allow_interruptions=True)
                logger.info(f"User inactive for {USER_INACTIVITY_TIMEOUT_SECONDS} seconds, say goodbye to the user and wait for {CLOSE_SESSION_AFTER_INACTIVITY_SECONDS} seconds before closing the session")

                await asyncio.sleep(CLOSE_SESSION_AFTER_INACTIVITY_SECONDS)  # Wait for 5 minutes before closing the session

                # Use _close_soon with USER_INACTIVITY reason
                logger.info(f"User inactive for {CLOSE_SESSION_AFTER_INACTIVITY_SECONDS} seconds after saying goodbye event, closing session")
                self.session._close_soon(reason=CloseReason.USER_INACTIVITY, drain=True)
                
            except asyncio.CancelledError:
                logger.debug("User inactivity timeout task cancelled")
                raise

        def on_user_state_changed(ev):
            logger.info(f"User state changed: {ev.old_state} -> {ev.new_state}")
            
            # Handle inactivity timeout logic
            if ev.new_state == "away":
                # User became inactive, start timeout timer
                if USER_INACTIVITY_TIMEOUT_SECONDS is not None:
                    # Cancel existing task if running
                    if self.inactivity_task is not None:
                        self.inactivity_task.cancel()
                    
                    self.inactivity_task = asyncio.create_task(inactive_user_timeout())
                    
                    # Add cleanup callback to prevent memory leaks
                    def cleanup_task(task):
                        self.inactivity_task = None
                    self.inactivity_task.add_done_callback(cleanup_task)
                    
                    logger.debug(f"Started inactivity timer for {USER_INACTIVITY_TIMEOUT_SECONDS} seconds")
            else:
                # User became active, cancel timeout timer if running
                if self.inactivity_task is not None:
                    self.inactivity_task.cancel()
                    self.inactivity_task = None
                    logger.debug("Cancelled inactivity timer - user became active")

        return on_user_state_changed

    def create_metrics_handler(self):
        """
        Create metrics collection event handler.

        Returns:
            Event handler function for metrics collection
        """

        def on_metrics_collected(ev: MetricsCollectedEvent):
            """Handle metrics collection events with comprehensive logging."""
            # Log detailed metrics
            metrics.log_metrics(ev.metrics)

            # Collect usage statistics
            if self.usage_collector:
                self.usage_collector.collect(ev.metrics)

            # Handle different types of metrics directly
            if isinstance(ev.metrics, metrics.EOUMetrics):
                self._handle_eou_metrics(ev.metrics)
            elif isinstance(ev.metrics, metrics.LLMMetrics):
                self._handle_llm_metrics(ev.metrics)
            elif isinstance(ev.metrics, metrics.TTSMetrics):
                self._handle_tts_metrics(ev.metrics)
            elif isinstance(ev.metrics, metrics.STFMetrics):
                self._handle_stf_metrics(ev.metrics)
            elif isinstance(ev.metrics, metrics.VADMetrics):
                self._handle_vad_metrics(ev.metrics)
            elif isinstance(ev.metrics, metrics.STTMetrics):
                self._handle_stt_metrics(ev.metrics)

        return on_metrics_collected


    def create_session_close_handler(self):
        """
        Create session close event handler.

        Returns:
            Event handler function for session closure
        """

        def on_session_close(ev: CloseEvent):
            """Handle session closure - save chat history, send RPC, and cleanup."""
            
            # Clean up inactivity task
            if self.inactivity_task is not None:
                self.inactivity_task.cancel()
                self.inactivity_task = None
            
            # Prevent duplicate session close handling
            if self._session_closing:
                logger.debug(f"Session already closing, ignoring close event: {ev.reason.value}")
                return
                
            # Additional connection check for safety
            if not self.ctx._connected:
                logger.debug(f"Context not connected, skipping session close handling: {ev.reason.value}")
                return
                
            # Mark session as closing
            self._session_closing = True
            logger.info(f"Session closing: {ev.reason.value}")
            
            # Log final usage statistics
            if self.usage_collector:
                summary = self.usage_collector.get_summary()
                logger.info(f"Session ended - Final usage statistics: {summary}")
            
            # Log database-stored usage summary with token info
            try:
                usage_summary = self.db.get_usage_summary(self.participant.identity)
                session_usage = self.db.get_session_usage(self.user_data.session_id)
                token_info = self.db.get_token_info(self.participant.identity)
                
                logger.info(
                    f"📊 Session Usage Summary:\n"
                    f"  - LLM Requests: {len(session_usage['llm_usage'])}\n" 
                    f"  - TTS Requests: {len(session_usage['tts_usage'])}\n"
                    f"  - Total LLM Tokens (All-time): {usage_summary['llm']['total_tokens']}\n"
                    f"  - Total TTS Characters (All-time): {usage_summary['tts']['total_characters']}\n"
                    f"  💰 Token Balance:\n"
                    f"    - Remaining: {token_info['remaining_tokens']}\n"
                    f"    - Used: {token_info['total_tokens_used']}\n"
                    f"    - Granted: {token_info['total_tokens_granted']}"
                )
            except Exception as e:
                logger.error(f"Failed to get usage summary from database: {e}")

            # Performance metrics were logged during execution

            # Extract chat messages from agent context (only new messages from current session)
            chat_messages = []
            start_index = self.agent._preloaded_message_count

            for item in self.agent.chat_ctx.items[start_index:]:
                if isinstance(item, llm.ChatMessage):
                    # Skip system messages
                    if item.role in ["system", "developer"]:
                        continue

                    # Convert timestamp to datetime
                    if isinstance(item.created_at, (int, float)):
                        timestamp = datetime.fromtimestamp(item.created_at)
                    elif isinstance(item.created_at, datetime):
                        timestamp = item.created_at
                    else:
                        timestamp = datetime.now()

                    # Extract content string
                    content_str = ""
                    if isinstance(item.content, list):
                        content_str = " ".join(str(c) for c in item.content)
                    else:
                        content_str = str(item.content)

                    chat_messages.append(
                        ChatMessage(
                            participant_id=self.participant.identity,
                            session_id=self.user_data.session_id,
                            timestamp=timestamp,
                            role=item.role,
                            content=content_str,
                            interrupted=getattr(item, "interrupted", False),
                        )
                    )

            logger.info(
                f"Session ended, saving chat history... (total {len(chat_messages)} messages)"
            )

            # Save to database
            if chat_messages:
                self.db.save_chat_messages(chat_messages)
                self.db.update_last_seen(self.participant.identity)

            # Log user summary
            user_summary = self.db.get_user_summary(self.participant.identity)
            logger.info(f"User summary: {user_summary}")
            
            # Prepare close reason details
            close_details = {
                "reason": ev.reason.value,
                "timestamp": time.time()
            }
            
            # Add specific details based on close reason
            if ev.reason == CloseReason.USER_INACTIVITY:
                close_details["timeout_seconds"] = USER_INACTIVITY_TIMEOUT_SECONDS
                close_details["detail"] = f"user_inactive_for_{USER_INACTIVITY_TIMEOUT_SECONDS}_seconds"
            
            # Send RPC notification about session closure
            try:
                payload = json.dumps(close_details)
                
                task = asyncio.create_task(
                    self.ctx.room.local_participant.perform_rpc(
                        destination_identity=self.participant.identity,
                        method="end_session",
                        payload=payload,
                        response_timeout=1.0
                    )
                )
                
                # Add completion callback for error logging
                def handle_rpc_result(future):
                    try:
                        future.result()
                        logger.debug(f"End session RPC sent successfully: {ev.reason.value}")
                    except Exception as e:
                        logger.warning(f"Failed to send end session RPC: {e}")

                task.add_done_callback(handle_rpc_result)
                
            except Exception as e:
                logger.error(f"Error sending session close RPC: {e}")
            
            # Safe room deletion with error handling
            # try:
            #     self.ctx.delete_room()
            #     logger.debug("Room deletion initiated successfully")
            # except Exception as e:
            #     # Log but don't re-raise - session cleanup should continue gracefully
            #     logger.warning(f"Room deletion failed (likely already deleted): {e}")

        return on_session_close


    def _handle_eou_metrics(self, eou_metrics: metrics.EOUMetrics):
        """Handle End-of-Utterance metrics."""
        # Store for E2E calculation
        self.last_eou_time = eou_metrics.last_speaking_time
        
        # Reset metrics collection for new cycle
        self.llm_ttft = None
        self.tts_ttfb = None
        self.stf_ttff = None
        self.e2e_latency = None
        self.metrics_logged = False
        
        # Log EOU components
        self.eou_ms = eou_metrics.end_of_utterance_delay * 1000
        self.stt_ms = eou_metrics.transcription_delay * 1000
        logger.debug(f"📝 EOU: {self.eou_ms:.0f}ms, STT: {self.stt_ms:.0f}ms")

    def _handle_llm_metrics(self, llm_metrics: metrics.LLMMetrics):
        """Handle LLM metrics and save to database."""
        self.llm_ttft = llm_metrics.ttft * 1000  # Convert to ms
        logger.debug(f"🧠 LLM TTFT: {self.llm_ttft:.0f}ms")
        
        # 데이터베이스에 실시간 저장
        try:
            # metrics 객체를 dict로 변환
            metrics_dict = {
                'request_id': llm_metrics.request_id,
                'prompt_tokens': llm_metrics.prompt_tokens,
                'prompt_cached_tokens': llm_metrics.prompt_cached_tokens,
                'completion_tokens': llm_metrics.completion_tokens,
                'total_tokens': llm_metrics.total_tokens,
                'duration': llm_metrics.duration,
                'cancelled': llm_metrics.cancelled,
                'label': llm_metrics.label,
                'ttft': llm_metrics.ttft,
                'tokens_per_second': llm_metrics.tokens_per_second,
                'speech_id': llm_metrics.speech_id
            }
            
            # DB에 저장
            self.db.save_llm_usage(
                participant_id=self.participant.identity,
                session_id=self.user_data.session_id,
                metrics=metrics_dict
            )
            
            # 실시간 사용량 로깅
            logger.info(
                f"💾 LLM Usage Saved - Tokens: {llm_metrics.total_tokens} "
                f"(prompt: {llm_metrics.prompt_tokens}, "
                f"cached: {llm_metrics.prompt_cached_tokens}, "
                f"completion: {llm_metrics.completion_tokens})"
            )
            
        except Exception as e:
            logger.error(f"Failed to save LLM usage to database: {e}")

    def _handle_tts_metrics(self, tts_metrics: metrics.TTSMetrics):
        """Handle TTS metrics and save to database."""
        self.tts_ttfb = tts_metrics.ttfb * 1000  # Convert to ms
        logger.debug(f"🔊 TTS TTFB: {self.tts_ttfb:.0f}ms")
        
        # 데이터베이스에 실시간 저장
        try:
            # metrics 객체를 dict로 변환
            metrics_dict = {
                'request_id': tts_metrics.request_id,
                'characters_count': tts_metrics.characters_count,
                'audio_duration': tts_metrics.audio_duration,
                'duration': tts_metrics.duration,
                'cancelled': tts_metrics.cancelled,
                'label': tts_metrics.label,
                'ttfb': tts_metrics.ttfb,
                'streamed': tts_metrics.streamed,
                'segment_id': tts_metrics.segment_id,
                'speech_id': tts_metrics.speech_id
            }
            
            # DB에 저장
            self.db.save_tts_usage(
                participant_id=self.participant.identity,
                session_id=self.user_data.session_id,
                metrics=metrics_dict
            )
            
            # 토큰 잔액 확인 및 로깅
            remaining_tokens = self.db.get_remaining_tokens(self.participant.identity)
            
            # 토큰 상태 체크 및 RPC 알림
            self._check_and_notify_token_status(remaining_tokens, tts_metrics.characters_count)
            
            # 실시간 사용량 로깅 (토큰 정보 포함)
            logger.info(
                f"💾 TTS Usage Saved - Characters: {tts_metrics.characters_count}, "
                f"Audio: {tts_metrics.audio_duration:.2f}s, "
                f"Remaining Tokens: {remaining_tokens}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save TTS usage to database: {e}")

    def _handle_stf_metrics(self, stf_metrics: metrics.STFMetrics):
        """Handle STF metrics and log complete pipeline."""
        # Record STF timing
        if hasattr(stf_metrics, 'ttff') and stf_metrics.ttff > 0:
            self.stf_ttff = stf_metrics.ttff * 1000  # Convert to ms
            logger.debug(f"🎭 STF TTFF: {self.stf_ttff:.0f}ms (streaming)")
        else:
            self.stf_ttff = stf_metrics.duration * 1000  # Fallback
            logger.debug(f"🎭 STF duration: {self.stf_ttff:.0f}ms (legacy)")
        
        # Log complete metrics once (STF is typically the last metric)
        if not self.metrics_logged:
            self._log_complete_metrics()
            self.metrics_logged = True

    def _handle_vad_metrics(self, vad_metrics: metrics.VADMetrics):
        """Handle VAD metrics (silent - too noisy for logs)."""
        _ = vad_metrics  # Acknowledge parameter to avoid linting warning
        pass

    def _handle_stt_metrics(self, stt_metrics: metrics.STTMetrics):
        """Handle STT metrics (silent - too noisy for logs)."""
        _ = stt_metrics  # Acknowledge parameter to avoid linting warning
        pass

    def _check_and_notify_token_status(self, remaining_tokens: int, characters_used: int):
        """토큰 상태 확인 및 필요시 RPC 알림 전송"""
        
        # 현재 상태 결정
        if remaining_tokens == 0:
            current_status = "depleted"
            message = "Your tokens are depleted"
        elif remaining_tokens <= self._token_thresholds["critical"]:
            current_status = "critical"
            message = f"Critical: Only {remaining_tokens} tokens remaining"
        elif remaining_tokens <= self._token_thresholds["low"]:
            current_status = "low"
            message = f"Low balance: {remaining_tokens} tokens remaining"
        else:
            current_status = "normal"
            message = f"Normal: {remaining_tokens} tokens remaining"
        
        # 상태 변경 감지 (악화된 경우만 알림)
        should_notify = False
        if current_status == "depleted" and self._last_token_status != "depleted":
            should_notify = True
        elif current_status == "critical" and self._last_token_status in ["low", "normal"]:
            should_notify = True
        elif current_status == "low" and self._last_token_status == "normal":
            should_notify = True
        
        # RPC 전송
        if should_notify:
            try:
                token_info = self.db.get_token_info(self.participant.identity)
                percentage = (remaining_tokens / token_info['total_tokens_granted'] * 100) if token_info['total_tokens_granted'] > 0 else 0
                
                payload = json.dumps({
                    "status": current_status,
                    "remaining_tokens": remaining_tokens,
                    "total_granted": token_info['total_tokens_granted'],
                    "total_used": token_info['total_tokens_used'],
                    "percentage_remaining": round(percentage, 1),
                    "last_usage": {
                        "characters": characters_used,
                        "timestamp": time.time()
                    },
                    "thresholds": self._token_thresholds,
                    "message": message
                })
                
                task = asyncio.create_task(
                    self.ctx.room.local_participant.perform_rpc(
                        destination_identity=self.participant.identity,
                        method="token_status_update",
                        payload=payload,
                        response_timeout=1.0
                    )
                )
                
                # Add completion callback for error logging
                def handle_rpc_result(future):
                    try:
                        future.result()
                        logger.debug(f"Token status RPC sent successfully: {current_status}")
                    except Exception as e:
                        logger.warning(f"Failed to send token status RPC: {e}")
                
                task.add_done_callback(handle_rpc_result)
                
                # 로깅
                logger.info(f"📢 Token status RPC sent: {current_status} - {message}")
                
            except Exception as e:
                logger.error(f"Error sending token status RPC: {e}")
            
            # 상태 업데이트
            self._last_token_status = current_status
    
    def _log_complete_metrics(self):
        """Log complete reactivity breakdown with E2E."""
        parts = []
        
        # Add E2E as the first metric if available
        if self.e2e_latency is not None:
            parts.append(f"E2E: {self.e2e_latency:.0f}ms")
        
        # Streaming latencies
        if self.stt_ms is not None:
            parts.append(f"STT: {self.stt_ms:.0f}ms")
        if self.llm_ttft is not None:
            parts.append(f"LLM: {self.llm_ttft:.0f}ms")
        if self.tts_ttfb is not None:
            parts.append(f"TTS: {self.tts_ttfb:.0f}ms")
        if self.stf_ttff is not None:
            parts.append(f"STF: {self.stf_ttff:.0f}ms")
        
        if parts:
            logger.info(f"🚀 Complete Metrics: {' | '.join(parts)}")
        
        # Save metrics to JSON file for allowed users
        if self.metrics_logging_enabled and self.metrics_output_dir:
            self._save_metrics_to_file()
    
    def _save_metrics_to_file(self):
        """Save complete metrics to JSON file for analysis."""
        try:
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds precision
            filename = f"{timestamp}_metrics.json"
            filepath = os.path.join(self.metrics_output_dir, filename)
            
            # Prepare metrics data
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "participant_identity": self.participant.identity,
                "agent_identity": getattr(self.ctx.room.local_participant, 'identity', 'unknown'),
                "session_id": getattr(self.user_data, 'session_id', 'unknown'),
                "metrics": {
                    "e2e_latency_ms": self.e2e_latency,
                    "stt_ms": self.stt_ms,
                    "llm_ttft_ms": self.llm_ttft,
                    "tts_ttfb_ms": self.tts_ttfb,
                    "stf_ttff_ms": self.stf_ttff,
                    "eou_ms": self.eou_ms
                },
                "context": {
                    "user_language": getattr(self.agent, 'user_language', 'unknown'),
                    "voice_name": getattr(self.agent, 'voice_name', 'unknown'),
                    "custom_persona": getattr(self.agent, 'custom_persona', '')
                }
            }
            
            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            # Log success
            logger.info(
                f"[METRICS SAVED] File: {filename}, "
                f"E2E: {self.e2e_latency:.0f}ms" if self.e2e_latency else f"[METRICS SAVED] File: {filename}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save metrics to {filepath}: {e}")
