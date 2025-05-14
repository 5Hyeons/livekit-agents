from __future__ import annotations

import asyncio
import time

from livekit import rtc

from ... import utils
from ...log import logger
from ...types import (
    ATTRIBUTE_TRANSCRIPTION_FINAL,
    ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID,
    ATTRIBUTE_TRANSCRIPTION_TRACK_ID,
    TOPIC_TRANSCRIPTION,
)
from .. import io
from ..transcription import find_micro_track_id
from ..avatar._animation_datastream_io import ANIMATION_STREAM_TOPIC


class _ParticipantAudioOutput(io.AudioOutput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        sample_rate: int,
        num_channels: int,
        track_publish_options: rtc.TrackPublishOptions,
        queue_size_ms: int = 100_000,  # TODO(long): move buffer to python
    ) -> None:
        super().__init__(next_in_chain=None, sample_rate=sample_rate)
        self._room = room
        self._lock = asyncio.Lock()
        self._audio_source = rtc.AudioSource(sample_rate, num_channels, queue_size_ms)
        self._publish_options = track_publish_options
        self._publication: rtc.LocalTrackPublication | None = None

        self._republish_task: asyncio.Task | None = None  # used to republish track on reconnection
        self._flush_task: asyncio.Task | None = None
        self._interrupted_event = asyncio.Event()

        self._pushed_duration: float = 0.0
        self._interrupted: bool = False

    async def _publish_track(self) -> None:
        async with self._lock:
            track = rtc.LocalAudioTrack.create_audio_track("roomio_audio", self._audio_source)
            self._publication = await self._room.local_participant.publish_track(
                track, self._publish_options
            )
            await self._publication.wait_for_subscription()

    async def start(self) -> None:
        await self._publish_track()

        def _on_reconnected() -> None:
            if self._republish_task:
                self._republish_task.cancel()
            self._republish_task = asyncio.create_task(self._publish_track())

        self._room.on("reconnected", _on_reconnected)

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

        if self._flush_task and not self._flush_task.done():
            logger.error("capture_frame called while flush is in progress")
            await self._flush_task

        self._pushed_duration += frame.duration
        await self._audio_source.capture_frame(frame)

    def flush(self) -> None:
        super().flush()

        if not self._pushed_duration:
            return

        if self._flush_task and not self._flush_task.done():
            # shouldn't happen if only one active speech handle at a time
            logger.error("flush called while playback is in progress")
            self._flush_task.cancel()

        self._flush_task = asyncio.create_task(self._wait_for_playout())

    def clear_buffer(self) -> None:
        super().clear_buffer()
        if not self._pushed_duration:
            return
        self._interrupted_event.set()

    async def _wait_for_playout(self) -> None:
        wait_for_interruption = asyncio.create_task(self._interrupted_event.wait())
        wait_for_playout = asyncio.create_task(self._audio_source.wait_for_playout())
        await asyncio.wait(
            [wait_for_playout, wait_for_interruption],
            return_when=asyncio.FIRST_COMPLETED,
        )

        interrupted = wait_for_interruption.done()
        pushed_duration = self._pushed_duration

        if interrupted:
            pushed_duration = max(pushed_duration - self._audio_source.queued_duration, 0)
            self._audio_source.clear_queue()
            wait_for_playout.cancel()
        else:
            wait_for_interruption.cancel()

        self._pushed_duration = 0
        self._interrupted_event.clear()
        self.on_playback_finished(playback_position=pushed_duration, interrupted=interrupted)


class _ParticipantLegacyTranscriptionOutput(io.TextOutput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        is_delta_stream: bool = True,
        participant: rtc.Participant | str | None = None,
    ):
        super().__init__(next_in_chain=None)
        self._room, self._is_delta_stream = room, is_delta_stream
        self._track_id: str | None = None
        self._participant_identity: str | None = None

        self._room.on("track_published", self._on_track_published)
        self._room.on("local_track_published", self._on_local_track_published)
        self._flush_task: asyncio.Task | None = None

        self._reset_state()
        self.set_participant(participant)

    def set_participant(
        self,
        participant: rtc.Participant | str | None,
    ) -> None:
        self._participant_identity = (
            participant.identity if isinstance(participant, rtc.Participant) else participant
        )
        if self._participant_identity is None:
            return

        try:
            self._track_id = find_micro_track_id(self._room, self._participant_identity)
        except ValueError:
            return

        self.flush()
        self._reset_state()

    def _reset_state(self) -> None:
        self._current_id = utils.shortuuid("SG_")
        self._capturing = False
        self._pushed_text = ""

    @utils.log_exceptions(logger=logger)
    async def capture_text(self, text: str) -> None:
        if self._participant_identity is None or self._track_id is None:
            return

        if self._flush_task and not self._flush_task.done():
            await self._flush_task

        if not self._capturing:
            self._reset_state()
            self._capturing = True

        if self._is_delta_stream:
            self._pushed_text += text
        else:
            self._pushed_text = text

        await self._publish_transcription(self._current_id, self._pushed_text, final=False)

    @utils.log_exceptions(logger=logger)
    def flush(self) -> None:
        if self._participant_identity is None or self._track_id is None or not self._capturing:
            return

        self._flush_task = asyncio.create_task(
            self._publish_transcription(self._current_id, self._pushed_text, final=True)
        )
        self._reset_state()

    async def _publish_transcription(self, id: str, text: str, final: bool) -> None:
        if self._participant_identity is None or self._track_id is None:
            return

        transcription = rtc.Transcription(
            participant_identity=self._participant_identity,
            track_sid=self._track_id,
            segments=[
                rtc.TranscriptionSegment(
                    id=id,
                    text=text,
                    start_time=0,
                    end_time=0,
                    final=final,
                    language="",
                )
            ],
        )
        try:
            if self._room.isconnected():
                await self._room.local_participant.publish_transcription(transcription)
        except Exception as e:
            logger.warning("failed to publish transcription", exc_info=e)

    def _on_track_published(
        self, track: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if (
            self._participant_identity is None
            or participant.identity != self._participant_identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        self._track_id = track.sid

    def _on_local_track_published(self, track: rtc.LocalTrackPublication, _: rtc.Track) -> None:
        if (
            self._participant_identity is None
            or self._participant_identity != self._room.local_participant.identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        self._track_id = track.sid


class _ParticipantTranscriptionOutput(io.TextOutput):
    def __init__(
        self,
        room: rtc.Room,
        *,
        is_delta_stream: bool = True,
        participant: rtc.Participant | str | None = None,
    ):
        super().__init__(next_in_chain=None)
        self._room, self._is_delta_stream = room, is_delta_stream
        self._track_id: str | None = None
        self._participant_identity: str | None = None

        self._writer: rtc.TextStreamWriter | None = None

        self._room.on("track_published", self._on_track_published)
        self._room.on("local_track_published", self._on_local_track_published)
        self._flush_atask: asyncio.Task | None = None

        self._reset_state()
        self.set_participant(participant)

    def set_participant(
        self,
        participant: rtc.Participant | str | None,
    ) -> None:
        self._participant_identity = (
            participant.identity if isinstance(participant, rtc.Participant) else participant
        )
        if self._participant_identity is None:
            return

        try:
            self._track_id = find_micro_track_id(self._room, self._participant_identity)
        except ValueError:
            # track id is optional for TextStream when audio is not published
            self._track_id = None

        self.flush()
        self._reset_state()

    def _reset_state(self) -> None:
        self._current_id = utils.shortuuid("SG_")
        self._capturing = False
        self._latest_text = ""

    async def _create_text_writer(
        self, attributes: dict[str, str] | None = None
    ) -> rtc.TextStreamWriter:
        assert self._participant_identity is not None, "participant_identity is not set"

        if not attributes:
            attributes = {
                ATTRIBUTE_TRANSCRIPTION_FINAL: "false",
            }
            if self._track_id:
                attributes[ATTRIBUTE_TRANSCRIPTION_TRACK_ID] = self._track_id
        attributes[ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID] = self._current_id

        return await self._room.local_participant.stream_text(
            topic=TOPIC_TRANSCRIPTION,
            sender_identity=self._participant_identity,
            attributes=attributes,
        )

    @utils.log_exceptions(logger=logger)
    async def capture_text(self, text: str) -> None:
        if self._participant_identity is None:
            return

        if self._flush_atask and not self._flush_atask.done():
            await self._flush_atask

        if not self._capturing:
            self._reset_state()
            self._capturing = True

        self._latest_text = text

        try:
            if self._room.isconnected():
                if self._is_delta_stream:  # reuse the existing writer
                    if self._writer is None:
                        self._writer = await self._create_text_writer()

                    await self._writer.write(text)
                else:  # always create a new writer
                    tmp_writer = await self._create_text_writer()
                    await tmp_writer.write(text)
                    await tmp_writer.aclose()
        except Exception as e:
            logger.warning("failed to publish transcription", exc_info=e)

    async def _flush_task(self, writer: rtc.TextStreamWriter | None):
        attributes = {
            ATTRIBUTE_TRANSCRIPTION_FINAL: "true",
        }
        if self._track_id:
            attributes[ATTRIBUTE_TRANSCRIPTION_TRACK_ID] = self._track_id

        try:
            if self._room.isconnected():
                if self._is_delta_stream:
                    if writer:
                        await writer.aclose(attributes=attributes)
                else:
                    tmp_writer = await self._create_text_writer(attributes=attributes)
                    await tmp_writer.write(self._latest_text)
                    await tmp_writer.aclose()
        except Exception as e:
            logger.warning("failed to publish transcription", exc_info=e)

    def flush(self) -> None:
        if self._participant_identity is None or not self._capturing:
            return

        self._capturing = False
        curr_writer = self._writer
        self._writer = None
        self._flush_atask = asyncio.create_task(self._flush_task(curr_writer))

    def _on_track_published(
        self, track: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if (
            self._participant_identity is None
            or participant.identity != self._participant_identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        self._track_id = track.sid

    def _on_local_track_published(self, track: rtc.LocalTrackPublication, _: rtc.Track) -> None:
        if (
            self._participant_identity is None
            or self._participant_identity != self._room.local_participant.identity
            or track.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        self._track_id = track.sid


class _ParticipantAnimationOutput(io.AnimationDataOutput):
    """
    애니메이션 데이터를 브로드캐스트하는 출력 클래스입니다.
    (수정됨: 스트림을 오랫동안 열어두는 단순화된 버전)
    """

    def __init__(
        self,
        room: rtc.Room,
        *,
        participant: rtc.Participant | str | None = None,
    ):
        super().__init__(next_in_chain=None)
        self._room = room
        self._participant_identity = (
            participant.identity if isinstance(participant, rtc.Participant) else participant
        )

        # self._current_stream: asyncio.Task | None = None # No longer needed?
        # self._pending_frames: list[io.AnimationData] = [] # No longer needed?
        self._close_task: asyncio.Task | None = None # Task for final close
        self._stream_writer: rtc.ByteStreamWriter | None = None
        self._frames_count = 0
        self._start_time = time.time()
        self._is_closed = False # Flag to indicate if closed

        # Removed self._segment_id

        logger.info(f"[ANIM_OUTPUT_SIMPLE] 초기화: 대상 참가자={self._participant_identity}")

    # Removed _reset_state method

    async def capture_frame(self, data: io.AnimationData) -> None:
        """애니메이션 프레임 데이터를 캡처합니다."""
        writer_id = id(self._stream_writer) if self._stream_writer else "None"
        # logger.debug(f"[ANIM_OUTPUT_SIMPLE] capture_frame 시작: Writer ID={writer_id}, 데이터 특징 개수={data.num_features}")

        if self._is_closed:
             logger.warning("[ANIM_OUTPUT_SIMPLE] capture_frame 호출됨, 그러나 이미 닫힘 상태임.")
             return

        if self._participant_identity is None:
            logger.warning("[ANIM_OUTPUT_SIMPLE] 대상 참가자가 없어 애니메이션 데이터를 전송할 수 없습니다.")
            return

        # Removed segment_id logic

        # 스트림 작성자가 없는 경우 생성 (최초 1회)
        if self._stream_writer is None:
            stream_id = utils.shortuuid("ANIMATION_")
            attributes = {
                "num_features": str(data.num_features),
                # segment_id is removed
            }
            logger.info(f"[ANIM_OUTPUT_SIMPLE] 새 애니메이션 데이터 스트림 생성 시도: 스트림 ID={stream_id}, 대상={self._participant_identity}, 특성 개수={data.num_features}, 속성={attributes}")
            try:
                self._stream_writer = await self._room.local_participant.stream_bytes(
                    name=stream_id,
                    topic=ANIMATION_STREAM_TOPIC,
                    destination_identities=[self._participant_identity] if self._participant_identity else None,
                    attributes=attributes,
                )
                self._frames_count = 0
                self._start_time = time.time()
                logger.info(f"[ANIM_OUTPUT_SIMPLE] 새 애니메이션 데이터 스트림 생성 성공: 스트림 ID={stream_id}, writer ID={id(self._stream_writer)}")
            except Exception as e:
                 logger.error(f"[ANIM_OUTPUT_SIMPLE] 애니메이션 데이터 스트림 생성 실패: {e}", exc_info=True)
                 self._stream_writer = None # Ensure writer is None if creation failed
                 return # Stop processing if stream creation failed

        # 애니메이션 데이터 쓰기
        if self._stream_writer is not None:
            try:
                data_size = len(data.data)
                writer_id = id(self._stream_writer)
                # logger.debug(f"[ANIM_OUTPUT_SIMPLE] 데이터 쓰기 시도: writer ID={writer_id}, 프레임 번호={self._frames_count + 1}, 데이터 크기={data_size}")
                await self._stream_writer.write(data.data)
                self._frames_count += 1

                if self._frames_count % 180 == 0:  # 약 3초분량 로깅
                    elapsed = time.time() - self._start_time
                    fps = self._frames_count / elapsed if elapsed > 0 else 0
                    logger.debug(f"[ANIM_OUTPUT_SIMPLE] 애니메이션 데이터 전송 중: {self._frames_count}개 프레임, 평균 FPS: {fps:.1f}, writer ID={writer_id}")

            except Exception as e:
                writer_id = id(self._stream_writer) if self._stream_writer else "None"
                logger.error(f"[ANIM_OUTPUT_SIMPLE] 애니메이션 데이터 전송 오류: writer ID={writer_id}, 오류={e}", exc_info=True)
                # Consider closing the stream permanently on write error?
                await self.close() # Attempt to close the stream if write fails

    async def close(self) -> None:
        """스트림을 명시적으로 닫습니다 (예: 에이전트 종료 시)."""
        if self._is_closed:
            logger.debug("[ANIM_OUTPUT_SIMPLE] close 호출됨, 이미 닫힘 상태.")
            return

        if self._stream_writer is None:
            logger.debug("[ANIM_OUTPUT_SIMPLE] close 호출됨, 그러나 writer가 없음.")
            self._is_closed = True
            return

        self._is_closed = True # Mark as closed immediately
        writer_to_close = self._stream_writer
        self._stream_writer = None # Prevent further writes
        writer_id = id(writer_to_close)
        frames_count = self._frames_count
        elapsed = time.time() - self._start_time
        fps = frames_count / elapsed if elapsed > 0 else 0

        logger.info(f"[ANIM_OUTPUT_SIMPLE] 스트림 닫기 시작 (close 호출): writer ID={writer_id}, 전송된 프레임={frames_count}, FPS={fps:.1f}")

        try:
            await writer_to_close.aclose()
            logger.info(f"[ANIM_OUTPUT_SIMPLE] 스트림 닫기 완료 (close 호출): writer ID={writer_id}")
        except Exception as e:
            logger.error(f"[ANIM_OUTPUT_SIMPLE] 애니메이션 스트림(ID: {writer_id}) 종료 중 오류 (close 호출): {e}", exc_info=True)


    # Removed _close_current_stream method as its logic is integrated into close()

    def flush(self) -> None:
        """(수정됨) 이 메서드는 더 이상 스트림을 닫지 않습니다."""
        writer_id = id(self._stream_writer) if self._stream_writer else "None"
        # logger.debug(f"[ANIM_OUTPUT_SIMPLE] flush 호출됨 (동작 없음): 현재 writer ID={writer_id}")
        # No operation needed for continuous animation stream in this simplified model
        pass

# Keep this utility private for now
class _ParallelTextOutput(io.TextOutput):
    def __init__(
        self, sinks: list[io.TextOutput], *, next_in_chain: io.TextOutput | None = None
    ) -> None:
        super().__init__(next_in_chain=next_in_chain)
        self._sinks = sinks

    async def capture_text(self, text: str) -> None:
        await asyncio.gather(*[sink.capture_text(text) for sink in self._sinks])

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()
