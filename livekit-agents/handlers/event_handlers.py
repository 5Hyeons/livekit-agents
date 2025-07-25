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

    def create_agent_state_handler(self):
        """
        Create agent state change event handler.

        Returns:
            Event handler function for agent state changes
        """

        def on_agent_state_changed(ev):
            """Handle agent state change events - track speaking state."""
            if ev.new_state == "speaking":
                # Record when agent starts speaking (correct timing)
                self.agent.reactivity_tracker.record_agent_utterance_start()

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
            """Handle user state change events - track speech end."""
            if ev.old_state == "speaking" and ev.new_state == "listening":
                # User finished speaking - start reactivity measurement
                self.agent.reactivity_tracker.record_user_speech_end()

            logger.info(f"User state changed: {ev.old_state} -> {ev.new_state}")

        return on_user_state_changed

    def create_metrics_handler(self):
        """
        Create metrics collection event handler.

        Returns:
            Event handler function for metrics collection
        """

        def on_metrics_collected(ev: MetricsCollectedEvent):
            """Handle metrics collection events."""
            # Log detailed metrics
            metrics.log_metrics(ev.metrics)

            # Collect usage statistics
            self.usage_collector.collect(ev.metrics)

            # Update agent's metrics data
            self.agent._update_metrics_data(ev.metrics)

        return on_metrics_collected

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

            # Log agent's performance summary
            self.agent._log_performance_summary()

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
