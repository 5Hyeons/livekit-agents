from __future__ import annotations

import asyncio
import json
import logging
import struct
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import asdict

import numpy as np
from livekit import rtc

from ... import utils
from ..io import AnimationDataOutput

# TYPE_CHECKING 임포트 추가
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...stf import AnimationData

logger = logging.getLogger(__name__)

ANIMATION_STREAM_TOPIC = "lk.animation_stream"
RPC_CLEAR_BUFFER = "lk.clear_buffer"  # 기존 오디오 클리어와 동일한 메서드 사용

# 애니메이션 데이터에 대한 메타데이터 형식
ANIMATION_DATA_HEADER_FORMAT = ">IIII"  # frame_index, num_features, timestamp_us, segment_id
ANIMATION_DATA_HEADER_SIZE = struct.calcsize(ANIMATION_DATA_HEADER_FORMAT)


class ByteStreamAnimationOutput(AnimationDataOutput):
    """
    AnimationDataOutput 구현체로, 애니메이션 데이터를 LiveKit DataStream을 통해 전송합니다.
    """

    def __init__(
        self, room: rtc.Room, *, destination_identity: str, topic: str = ANIMATION_STREAM_TOPIC
    ):
        super().__init__(next_in_chain=None)
        self._room = room
        self._destination_identity = destination_identity
        self._topic = topic
        self._stream_writer: rtc.ByteStreamWriter | None = None
        self._frame_index = 0
        self._tasks: set[asyncio.Task] = set()

    async def capture_frame(self, data: "AnimationData") -> None:
        """애니메이션 프레임을 캡처하고 대상에게 스트리밍합니다."""
        
        if not self._stream_writer:
            self._stream_writer = await self._room.local_participant.stream_bytes(
                name=utils.shortuuid("ANIMATION_"),
                topic=self._topic,
                destination_identities=[self._destination_identity],
                attributes={
                    "num_features": str(data.num_features),
                    "segment_id": str(data.segment_id),
                },
            )
            self._frame_index = 0
        
        # 프레임 헤더 생성 (프레임 인덱스, 특성 수, 타임스탬프)
        header = struct.pack(
            ANIMATION_DATA_HEADER_FORMAT, 
            self._frame_index, 
            data.num_features, 
            data.timestamp_us,
            data.segment_id
        )
        
        # 헤더와 데이터를 함께 전송
        await self._stream_writer.write(header + data.data)
        self._frame_index += 1

    def flush(self) -> None:
        """현재 애니메이션 세그먼트의 끝을 표시합니다."""
        if self._stream_writer is None:
            return

        # 스트림을 닫아 세그먼트 끝 표시
        task = asyncio.create_task(self._stream_writer.aclose())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        self._stream_writer = None
        logger.debug(
            "animation data stream flushed",
            extra={"frame_count": self._frame_index},
        )
        self._frame_index = 0

    def clear_buffer(self) -> None:
        """버퍼를 지우고 즉시 재생을 중지합니다."""
        task = asyncio.create_task(
            self._room.local_participant.perform_rpc(
                destination_identity=self._destination_identity,
                method=RPC_CLEAR_BUFFER,
                payload="",
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        """자원을 정리합니다."""
        if self._stream_writer:
            await self._stream_writer.aclose()
            self._stream_writer = None
        
        # 모든 대기 중인 작업 취소
        await utils.aio.cancel_and_wait(*self._tasks)
        self._tasks.clear() 