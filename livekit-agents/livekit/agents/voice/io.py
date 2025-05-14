from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Awaitable
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union, TYPE_CHECKING

from livekit import rtc

from .. import llm, stt
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr

if TYPE_CHECKING:
    from .agent import ModelSettings

# TODO(theomonnom): can those types be simplified?
STTNode = Callable[
    [AsyncIterable[rtc.AudioFrame], "ModelSettings"],
    Union[
        Optional[Union[AsyncIterable[stt.SpeechEvent], AsyncIterable[str]]],
        Awaitable[Optional[Union[AsyncIterable[stt.SpeechEvent], AsyncIterable[str]]]],
    ],
]
LLMNode = Callable[
    [llm.ChatContext, list[llm.FunctionTool], "ModelSettings"],
    Union[
        Optional[Union[AsyncIterable[llm.ChatChunk], AsyncIterable[str], str]],
        Awaitable[Optional[Union[AsyncIterable[llm.ChatChunk], AsyncIterable[str], str]]],
    ],
]
TTSNode = Callable[
    [AsyncIterable[str], "ModelSettings"],
    Union[
        Optional[AsyncIterable[rtc.AudioFrame]],
        Awaitable[Optional[AsyncIterable[rtc.AudioFrame]]],
    ],
]


class TimedString(str):
    start_time: NotGivenOr[float]
    end_time: NotGivenOr[float]

    def __new__(
        cls,
        text: str,
        start_time: NotGivenOr[float] = NOT_GIVEN,
        end_time: NotGivenOr[float] = NOT_GIVEN,
    ) -> TimedString:
        obj = super().__new__(cls, text)
        obj.start_time = start_time
        obj.end_time = end_time
        return obj


class AudioInput:
    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame: ...

    def on_attached(self) -> None: ...

    def on_detached(self) -> None: ...


class VideoInput:
    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame]:
        return self

    async def __anext__(self) -> rtc.VideoFrame: ...

    def on_attached(self) -> None: ...

    def on_detached(self) -> None: ...


@dataclass
class PlaybackFinishedEvent:
    playback_position: float
    """How much of the audio was played back"""
    interrupted: bool
    """Interrupted is True if playback was interrupted (clear_buffer() was called)"""
    synchronized_transcript: str | None = None
    """Transcript synced with playback; may be partial if the audio was interrupted
    When None, the transcript is not synchronized with the playback"""


class AudioOutput(ABC, rtc.EventEmitter[Literal["playback_finished"]]):
    def __init__(
        self,
        *,
        next_in_chain: AudioOutput | None = None,
        sample_rate: int | None = None,
    ) -> None:
        """
        Args:
            sample_rate: The sample rate required by the audio sink, if None, any sample rate is accepted
        """  # noqa: E501
        super().__init__()
        self._next_in_chain = next_in_chain
        self._sample_rate = sample_rate
        self.__capturing = False
        self.__playback_finished_event = asyncio.Event()

        self.__playback_segments_count = 0
        self.__playback_finished_count = 0
        self.__last_playback_ev: PlaybackFinishedEvent = PlaybackFinishedEvent(
            playback_position=0, interrupted=False
        )

        if self._next_in_chain:
            self._next_in_chain.on(
                "playback_finished",
                lambda ev: self.on_playback_finished(
                    interrupted=ev.interrupted,
                    playback_position=ev.playback_position,
                    synchronized_transcript=ev.synchronized_transcript,
                ),
            )

    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        """
        Developers building audio sinks must call this method when a playback/segment is finished.
        Segments are segmented by calls to flush() or clear_buffer()
        """

        if self.__playback_finished_count >= self.__playback_segments_count:
            logger.warning(
                "playback_finished called more times than playback segments were captured"
            )
            return

        self.__playback_finished_count += 1
        self.__playback_finished_event.set()

        ev = PlaybackFinishedEvent(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )
        self.__last_playback_ev = ev
        self.emit("playback_finished", ev)

    async def wait_for_playout(self) -> PlaybackFinishedEvent:
        """
        Wait for the past audio segments to finish playing out.

        Returns:
            PlaybackFinishedEvent: The event that was emitted when the audio finished playing out
            (only the last segment information)
        """
        target = self.__playback_segments_count

        while self.__playback_finished_count < target:
            await self.__playback_finished_event.wait()
            self.__playback_finished_event.clear()

        return self.__last_playback_ev

    @property
    def sample_rate(self) -> int | None:
        """The sample rate required by the audio sink, if None, any sample rate is accepted"""
        return self._sample_rate

    @abstractmethod
    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture an audio frame for playback, frames can be pushed faster than real-time"""
        if not self.__capturing:
            self.__capturing = True
            self.__playback_segments_count += 1

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered audio, marking the current playback/segment as complete"""
        self.__capturing = False

    @abstractmethod
    def clear_buffer(self) -> None:
        """Clear the buffer, stopping playback immediately"""

    def on_attached(self) -> None:
        if self._next_in_chain:
            self._next_in_chain.on_attached()

    def on_detached(self) -> None:
        if self._next_in_chain:
            self._next_in_chain.on_detached()


class TextOutput(ABC):
    def __init__(self, *, next_in_chain: TextOutput | None) -> None:
        self._next_in_chain = next_in_chain

    @abstractmethod
    async def capture_text(self, text: str) -> None:
        """Capture a text segment (Used by the output of LLM nodes)"""

    @abstractmethod
    def flush(self) -> None:
        """Mark the current text segment as complete (e.g LLM generation is complete)"""

    def on_attached(self) -> None:
        if self._next_in_chain:
            self._next_in_chain.on_attached()

    def on_detached(self) -> None:
        if self._next_in_chain:
            self._next_in_chain.on_detached()


# TODO(theomonnom): Add documentation to VideoSink
class VideoOutput(ABC):
    def __init__(self, *, next_in_chain: VideoOutput | None) -> None:
        self._next_in_chain = next_in_chain

    @abstractmethod
    async def capture_frame(self, text: rtc.VideoFrame) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    def on_attached(self) -> None:
        if self._next_in_chain:
            self._next_in_chain.on_attached()

    def on_detached(self) -> None:
        if self._next_in_chain:
            self._next_in_chain.on_detached()


# AnimationData는 STF 블렌드쉐입 데이터로, NumPy 배열을 직렬화해서 전송하기 위한 래퍼 클래스입니다.
class AnimationData:
    """애니메이션 데이터 클래스 - 얼굴 블렌드쉐입 데이터를 나타냅니다."""
    
    def __init__(self, data: bytes, num_features: int = 52, timestamp_us: int = 0, segment_id: str = "") -> None:
        self.data = data  # 직렬화된 블렌드쉐입 데이터
        self.num_features = num_features  # 블렌드쉐입 특성의 개수
        self.timestamp_us = timestamp_us  # 마이크로초 단위의 타임스탬프
        self.segment_id = segment_id  # 세그먼트 ID
    
    @classmethod
    def from_numpy(cls, arr, timestamp_us: int = 0, segment_id: str = ""):
        """NumPy 배열에서 AnimationData 객체 생성"""
        import numpy as np
        # NumPy 배열을 직렬화
        data = np.array(arr, dtype=np.float32).tobytes()
        return cls(data=data, num_features=len(arr), timestamp_us=timestamp_us, segment_id=segment_id)


# AnimationDataOutput은 RTF에서 생성된 블렌드쉐입 애니메이션 데이터를 처리하기 위한 인터페이스입니다.
class AnimationDataOutput(ABC):
    """STF(Speech-To-Face)에서 생성된 애니메이션 데이터를 처리하는 추상 클래스입니다."""
    
    def __init__(self, *, next_in_chain: 'AnimationDataOutput' | None = None) -> None:
        self._next_in_chain = next_in_chain
    
    @abstractmethod
    async def capture_frame(self, data: AnimationData) -> None:
        """애니메이션 프레임 데이터를 캡처합니다."""
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """현재 데이터 스트림을 플러시합니다."""
        pass
    
    def on_attached(self) -> None:
        """출력이 연결될 때 호출됩니다."""
        if self._next_in_chain:
            self._next_in_chain.on_attached()
    
    def on_detached(self) -> None:
        """출력이 분리될 때 호출됩니다."""
        if self._next_in_chain:
            self._next_in_chain.on_detached()


class AgentInput:
    def __init__(self, video_changed: Callable, audio_changed: Callable) -> None:
        self._video_stream: VideoInput | None = None
        self._audio_stream: AudioInput | None = None
        self._video_changed = video_changed
        self._audio_changed = audio_changed

        # enabled by default
        self._audio_enabled = True
        self._video_enabled = True

    def set_audio_enabled(self, enable: bool):
        if enable == self._audio_enabled:
            return

        self._audio_enabled = enable

        if not self._audio_stream:
            return

        if enable:
            self._audio_stream.on_attached()
        else:
            self._audio_stream.on_detached()

    def set_video_enabled(self, enable: bool):
        if enable == self._video_enabled:
            return

        self._video_enabled = enable

        if not self._video_stream:
            return

        if enable:
            self._video_stream.on_attached()
        else:
            self._video_stream.on_detached()

    @property
    def audio_enabled(self) -> bool:
        return self._audio_enabled

    @property
    def video_enabled(self) -> bool:
        return self._video_enabled

    @property
    def video(self) -> VideoInput | None:
        return self._video_stream

    @video.setter
    def video(self, stream: VideoInput | None) -> None:
        self._video_stream = stream
        self._video_changed()

    @property
    def audio(self) -> AudioInput | None:
        return self._audio_stream

    @audio.setter
    def audio(self, stream: AudioInput | None) -> None:
        self._audio_stream = stream
        self._audio_changed()


class AgentOutput:
    def __init__(
        self,
        video_changed: Callable,
        audio_changed: Callable,
        transcription_changed: Callable,
        animation_changed: Callable = lambda: None,  # 기본값으로 빈 콜백 추가
    ) -> None:
        self._video_sink: VideoOutput | None = None
        self._audio_sink: AudioOutput | None = None
        self._transcription_sink: TextOutput | None = None
        self._animation_sink: AnimationDataOutput | None = None
        self._video_changed = video_changed
        self._audio_changed = audio_changed
        self._transcription_changed = transcription_changed
        self._animation_changed = animation_changed

        self._audio_enabled = True
        self._video_enabled = True
        self._transcription_enabled = True
        self._animation_enabled = True

    def set_video_enabled(self, enabled: bool):
        if enabled == self._video_enabled:
            return

        self._video_enabled = enabled

        if not self._video_sink:
            return

        if enabled:
            self._video_sink.on_attached()
        else:
            self._video_sink.on_detached()

    def set_audio_enabled(self, enabled: bool):
        if enabled == self._audio_enabled:
            return

        self._audio_enabled = enabled

        if not self._audio_sink:
            return

        if enabled:
            self._audio_sink.on_attached()
        else:
            self._audio_sink.on_detached()

    def set_transcription_enabled(self, enabled: bool):
        if enabled == self._transcription_enabled:
            return

        self._transcription_enabled = enabled

        if not self._transcription_sink:
            return

        if enabled:
            self._transcription_sink.on_attached()
        else:
            self._transcription_sink.on_detached()

    def set_animation_enabled(self, enabled: bool):
        """애니메이션 데이터 출력 활성화/비활성화"""
        if enabled == self._animation_enabled:
            return

        self._animation_enabled = enabled

        if not self._animation_sink:
            return

        if enabled:
            self._animation_sink.on_attached()
        else:
            self._animation_sink.on_detached()

    @property
    def audio_enabled(self) -> bool:
        return self._audio_enabled

    @property
    def video_enabled(self) -> bool:
        return self._video_enabled

    @property
    def transcription_enabled(self) -> bool:
        return self._transcription_enabled

    @property
    def animation_enabled(self) -> bool:
        """애니메이션 데이터 출력 활성화 상태"""
        return self._animation_enabled

    @property
    def video(self) -> VideoOutput | None:
        return self._video_sink

    @video.setter
    def video(self, sink: VideoOutput | None) -> None:
        self._video_sink = sink
        self._video_changed()

    @property
    def audio(self) -> AudioOutput | None:
        return self._audio_sink

    @audio.setter
    def audio(self, sink: AudioOutput | None) -> None:
        if sink is self._audio_sink:
            return

        if self._audio_sink:
            self._audio_sink.on_detached()

        self._audio_sink = sink
        self._audio_changed()

        if self._audio_sink:
            self._audio_sink.on_attached()

    @property
    def transcription(self) -> TextOutput | None:
        return self._transcription_sink

    @transcription.setter
    def transcription(self, sink: TextOutput | None) -> None:
        if sink is self._transcription_sink:
            return

        if self._transcription_sink:
            self._transcription_sink.on_detached()

        self._transcription_sink = sink
        self._transcription_changed()

        if self._transcription_sink:
            self._transcription_sink.on_attached()

    @property
    def animation(self) -> AnimationDataOutput | None:
        """애니메이션 데이터 출력 싱크"""
        return self._animation_sink

    @animation.setter
    def animation(self, sink: AnimationDataOutput | None) -> None:
        """애니메이션 데이터 출력 싱크 설정"""
        if sink is self._animation_sink:
            return

        if self._animation_sink:
            self._animation_sink.on_detached()

        self._animation_sink = sink
        self._animation_changed()

        if self._animation_sink:
            self._animation_sink.on_attached()
