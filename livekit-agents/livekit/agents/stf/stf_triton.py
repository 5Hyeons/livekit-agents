from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import AsyncIterable, AsyncIterator
from typing import Protocol, Optional

import tritonclient.http as httpclient
import librosa
import numpy as np
from livekit import rtc
import time
import uuid

from ..log import logger
from ..utils import aio
from ..voice.io import AnimationData


class STFTritonStream:
    """Speech-to-Face stream class using Triton Inference Server."""

    def __init__(self, stf: "FaceAnimatorSTFTriton", chunk_duration_sec: float) -> None:
        self._stf = stf
        self._audio_queue = asyncio.Queue[Optional[rtc.AudioFrame]]()
        self._is_closed = False
        self._task: Optional[asyncio.Task] = None
        self._blendshape_frames = asyncio.Queue[Optional[np.ndarray]]()
        self._last_frame_time = 0.0
        self._chunk_duration_sec = chunk_duration_sec

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Add audio frame to the STF processing queue."""
        if self._is_closed:
            raise RuntimeError("STFTritonStream is closed")
        self._audio_queue.put_nowait(frame)

    def end_input(self) -> None:
        """Signal that audio input is complete."""
        self._audio_queue.put_nowait(None)

    def flush(self) -> None:
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()
        self.end_input()

    async def _send_triton_inference_request(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Send inference request to Triton server."""
        try:
            # Prepare audio input (1차원 float32 배열)
            audio_input = audio_data.astype(np.float32)
            # Audio scalar - using default value from tm_test.py
            audio_scalar = np.array([1.2], dtype=np.float32)
            
            # Create Triton input objects
            input_audio = httpclient.InferInput("audio_input", audio_input.shape, "FP32")
            input_scalar = httpclient.InferInput("audio_scalar", audio_scalar.shape, "FP32")
            input_audio.set_data_from_numpy(audio_input)
            input_scalar.set_data_from_numpy(audio_scalar)
            
            # Send inference request
            response = self._stf._client.infer(
                self._stf._model_name,
                inputs=[input_audio, input_scalar],
            )
            
            # Get output - shape is (1, num_frames, 52)
            output = response.as_numpy("anim_output")
            logger.debug(f"Triton server response shape: {output.shape}")
            
            return output
            
        except Exception as e:
            logger.error(f"Error during Triton inference request: {e}", exc_info=True)
            return None

    async def _process_frames(self) -> None:
        """Process audio frames and get blendshape data from Triton server."""
        audio_buffer = np.array([], dtype=np.float32)
        min_samples = int(self._chunk_duration_sec * 16000)  # Process in chunks
        frames_processed = 0
        animations_generated = 0
        start_time = time.time()

        logger.info(f"STF Triton 프레임 처리 시작 (서버: {self._stf._triton_url}, 모델: {self._stf._model_name}, 청크 크기: {self._chunk_duration_sec}초)")

        try:
            while True:
                frame = await self._audio_queue.get()

                if frame is None:
                    logger.debug("STF Triton 입력 종료 신호 수신")
                    break

                frames_processed += 1

                # Convert frame data to float32
                frame_data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)

                # Resample if necessary (target 16kHz)
                if frame.sample_rate != 16000:
                    try:
                        frame_data = librosa.resample(
                            frame_data, orig_sr=frame.sample_rate, target_sr=16000
                        )
                    except Exception as e:
                        logger.error(f"Error during audio resampling: {e}", exc_info=True)
                        continue

                # Add to buffer
                audio_buffer = np.concatenate((audio_buffer, frame_data))

                # Process audio in chunks
                while len(audio_buffer) >= min_samples:
                    audio_to_process = audio_buffer[:min_samples]
                    # Keep remaining audio
                    audio_buffer = audio_buffer[min_samples:]

                    logger.debug(f"STF Triton 처리 청크 준비 (길이: {len(audio_to_process)/16000.0:.2f}초)")
                    
                    # Send request to Triton server
                    animation_output = await self._send_triton_inference_request(audio_to_process)

                    if animation_output is not None and animation_output.size > 0:
                        # Output shape is (1, num_frames, 52) - need to flatten first dimension
                        # Remove the first dimension and iterate over frames
                        flattened_output = animation_output[0]  # Shape becomes (num_frames, 52)
                        num_frames = flattened_output.shape[0]
                        logger.debug(f"Triton 서버로부터 {num_frames}개 블렌드쉐입 프레임 수신")
                        
                        # Put each frame (numpy array) into the queue
                        for blendshape_frame in flattened_output:
                            await self._blendshape_frames.put(blendshape_frame)
                            animations_generated += 1

            # Process any remaining audio in the buffer after input ends
            if len(audio_buffer) > 0:
                buffer_duration = len(audio_buffer) / 16000.0
                logger.debug(f"남은 오디오 처리: {buffer_duration:.2f}초 ({len(audio_buffer)} 샘플)")

                animation_output = await self._send_triton_inference_request(audio_buffer)
                if animation_output is not None and animation_output.size > 0:
                    flattened_output = animation_output[0]
                    num_frames = flattened_output.shape[0]
                    logger.debug(f"최종 STF Triton 추론: 서버로부터 {num_frames}개 프레임 수신")
                    for blendshape_frame in flattened_output:
                        await self._blendshape_frames.put(blendshape_frame)
                        animations_generated += 1

        except Exception as e:
            logger.error(f"STF Triton 프레임 처리 중 오류 발생: {e}", exc_info=True)
        finally:
            duration = time.time() - start_time
            logger.info(
                f"STF Triton 프레임 처리 완료: {frames_processed}개 오디오 프레임 처리, "
                f"{animations_generated}개 애니메이션 생성, 총 소요 시간: {duration:.2f}초"
            )
            # Use None as the end marker
            await self._blendshape_frames.put(None)

    def __aiter__(self) -> AsyncIterator[AnimationData]:
        return self

    async def __anext__(self) -> AnimationData:
        """Return animation data."""
        if self._task is None:
            logger.debug("STF Triton 프레임 처리 태스크 시작")
            self._task = asyncio.create_task(self._process_frames())
            self._last_frame_time = asyncio.get_event_loop().time()

        blendshape_frame = await self._blendshape_frames.get()
        if blendshape_frame is None:
            logger.debug("STF Triton 스트림 종료")
            # Ensure task is finished before raising StopAsyncIteration
            if self._task and not self._task.done():
                try:
                    await self._task
                except Exception:
                    logger.error("Error during final task wait in STFTritonStream.__anext__", exc_info=True)
            raise StopAsyncIteration

        # Create timestamp
        timestamp_us = int(asyncio.get_event_loop().time() * 1_000_000)
        segment_id = str(uuid.uuid4())

        # Create AnimationData object directly from the blendshape numpy array
        animation_data = AnimationData.from_numpy(
            blendshape_frame, timestamp_us=timestamp_us, segment_id=segment_id
        )

        return animation_data

    async def aclose(self) -> None:
        """Close the stream."""
        if not self._is_closed:
            logger.debug("STF Triton 스트림 aclose 호출")
            self.flush()
            self._is_closed = True
            await aio.cancel_and_wait(self._task)

    async def __aenter__(self) -> "STFTritonStream":
        """Enter asynchronous context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit asynchronous context and clean up resources."""
        await self.aclose()


class STFTriton(Protocol):
    """Abstract base class for Speech-To-Face implementations using Triton."""

    async def synthesize(
        self, audio_stream: AsyncIterable[rtc.AudioFrame]
    ) -> AsyncIterable[AnimationData]:
        """Generate animation data from an audio stream."""
        ...
        if False:
            yield

    def stream(self) -> STFTritonStream:
        """Create a new STF stream for frame-by-frame processing."""
        ...

    async def aclose(self) -> None:
        """Close the STF component and release resources."""
        pass


class FaceAnimatorSTFTriton(STFTriton):
    """Implementation using Triton Inference Server."""

    def __init__(
        self,
        *,
        triton_url: str = "61.14.209.9:8400",
        model_name: str = "ensemble_model",
        frame_rate: int = 60,
        sample_rate: int = 16000,
        num_features: int = 52,
        chunk_duration_sec: float = 1.0,
    ) -> None:
        self._triton_url = triton_url
        self._model_name = model_name
        self._sample_rate = sample_rate
        self._frame_rate = frame_rate
        self._num_features = num_features
        self._chunk_duration_sec = chunk_duration_sec

        # Create Triton HTTP client
        self._client = httpclient.InferenceServerClient(url=triton_url)
        
        # Keep track of active streams
        self._streams = weakref.WeakSet[STFTritonStream]()

        logger.info(f"Initialized FaceAnimatorSTFTriton client for server: {triton_url}, model: {model_name}")

    async def aclose(self) -> None:
        """Close the FaceAnimatorSTFTriton client and associated streams."""
        logger.debug("Closing FaceAnimatorSTFTriton client and streams.")
        # Close all active streams associated with this instance
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

    def stream(self) -> STFTritonStream:
        """Create a new STF stream."""
        stream = STFTritonStream(self, chunk_duration_sec=self._chunk_duration_sec)
        # Track the created stream
        self._streams.add(stream)
        return stream

    async def synthesize(
        self, audio_stream: AsyncIterable[rtc.AudioFrame]
    ) -> AsyncIterable[AnimationData]:
        """
        Generate face animation data from an audio stream using Triton server.
        Maintained for compatibility.
        """
        async with self.stream() as stream:
            # Forward audio frames to the stream
            forward_task = asyncio.create_task(
                self._forward_audio_to_stream(audio_stream, stream)
            )
            try:
                # Yield the generated animation data
                async for data in stream:
                    yield data
            finally:
                # Ensure the forwarding task is cancelled on exit
                await aio.gracefully_cancel(forward_task)

    async def _forward_audio_to_stream(
        self, audio_stream: AsyncIterable[rtc.AudioFrame], stream: STFTritonStream
    ) -> None:
        """Forwards audio frames from an iterable to the STF stream."""
        try:
            async for frame in audio_stream:
                stream.push_frame(frame)
        except Exception as e:
            logger.error(f"Error forwarding audio to STF Triton stream: {e}", exc_info=True)
        finally:
            # Signal end of audio input
            stream.end_input() 