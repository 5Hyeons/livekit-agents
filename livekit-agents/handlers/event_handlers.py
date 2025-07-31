"""
Event handlers for agent session management.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from user_database import ChatMessage, UserData, UserDatabase

from livekit import rtc
from livekit.agents import llm, metrics
from livekit.agents.voice import MetricsCollectedEvent

if TYPE_CHECKING:
    from agent.wallmate_agent import WallmateAgent

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
        agent: "WallmateAgent",
        participant: rtc.RemoteParticipant,
        ctx_room: rtc.Room,
        db: UserDatabase,
        user_data: UserData,
        usage_collector: metrics.UsageCollector,
    ):
        """
        Initialize event handlers with required dependencies.

        Args:
            agent: WallmateAgent instance
            participant: Remote participant
            ctx_room: Room context
            db: Database instance
            user_data: User data
            usage_collector: Metrics collector
        """
        self.agent = agent
        self.participant = participant
        self.ctx_room = ctx_room
        self.db = db
        self.user_data = user_data
        self.usage_collector = usage_collector
        
        # E2E metrics tracking
        self.last_eou_time = None  # Track user speech end time
        
        self.eou_ms = None
        self.stt_ms = None
        self.llm_ttft = None  # LLM Time to First Token
        self.tts_ttfb = None  # TTS Time to First Byte
        self.stf_ttff = None  # STF Time to First Frame
        self.e2e_latency = None  # E2E Response Time
        self.metrics_logged = False  # Prevent duplicate logging

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
                    self.ctx_room.local_participant.perform_rpc(
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

        def on_user_state_changed(ev):
            logger.info(f"User state changed: {ev.old_state} -> {ev.new_state}")

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
        """Handle LLM metrics."""
        self.llm_ttft = llm_metrics.ttft * 1000  # Convert to ms
        logger.debug(f"🧠 LLM TTFT: {self.llm_ttft:.0f}ms")

    def _handle_tts_metrics(self, tts_metrics: metrics.TTSMetrics):
        """Handle TTS metrics."""
        self.tts_ttfb = tts_metrics.ttfb * 1000  # Convert to ms
        logger.debug(f"🔊 TTS TTFB: {self.tts_ttfb:.0f}ms")

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
        pass

    def _handle_stt_metrics(self, stt_metrics: metrics.STTMetrics):
        """Handle STT metrics (silent - too noisy for logs)."""
        pass

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

    def create_session_close_handler(self):
        """
        Create session close event handler.

        Returns:
            Event handler function for session closure
        """

        def on_session_close():
            """Handle session closure - save chat history and cleanup."""

            # Log final usage statistics
            summary = self.usage_collector.get_summary()
            logger.info(f"Session ended - Final usage statistics: {summary}")

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

        return on_session_close
