from __future__ import annotations

import asyncio
import time
import uuid
import weakref
from collections.abc import AsyncIterable, AsyncIterator
from enum import Enum
from typing import Literal, Optional

from abc import ABC
from typing import Generic, Literal, TypeVar, Union


import librosa
import numpy as np
import tritonclient.http as httpclient

from livekit import rtc

from ..log import logger
from ..metrics.base import STFMetrics
from ..utils import aio
from .animation_data import AnimationData


class OutputMode(str, Enum):
    """Output mode for FaceAnimator."""
    ANIMATION_WITH_AUDIO = "animation_with_audio"  # Default: outputs (animation, audio) pairs
    ANIMATION_ONLY = "animation_only"  # Legacy: outputs animation data only

TEvent = TypeVar("TEvent")

class STF(
    ABC,
    rtc.EventEmitter[Union[Literal["metrics_collected", "error"], TEvent]],
    Generic[TEvent],):
    """Abstract base class for Speech-To-Face implementations."""

    def __init__(self) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        return self._label

    async def synthesize(
        self, audio_stream: AsyncIterable[rtc.AudioFrame]
    ) -> AsyncIterable[AnimationData]:
        """Generate animation data from an audio stream."""
        ...

    def stream(self) -> "FaceAnimatorStream":
        """Create a new STF stream for frame-by-frame processing."""
        ...

    async def aclose(self) -> None:
        """Close the STF component and release resources."""
        pass


class FaceAnimatorStream:
    """Unified Speech-to-Face stream class supporting multiple output modes."""

    def __init__(
        self, 
        face_animator: "FaceAnimator", 
        chunk_duration_sec: float,
        output_mode: OutputMode = OutputMode.ANIMATION_WITH_AUDIO
    ) -> None:
        self._face_animator = face_animator
        self._audio_queue = asyncio.Queue[Optional[rtc.AudioFrame]]()
        self._is_closed = False
        self._task: asyncio.Task | None = None
        self._output_mode = output_mode
        self._chunk_duration_sec = chunk_duration_sec
        self._last_frame_time = 0.0
        
        # Output queue - type depends on output mode
        if output_mode == OutputMode.ANIMATION_ONLY:
            self._output_queue = asyncio.Queue[Optional[np.ndarray]]()
        else:
            # (animation, audio_samples, sample_rate, channels)
            self._output_queue = asyncio.Queue[Optional[tuple[np.ndarray, np.ndarray, int, int]]]()
        
        # Metrics tracking for streaming mode
        self._started_time: float = 0
        self._request_id: str = ""
        self._first_output_recorded: bool = False
        self._ttff: float = 0.0

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Add audio frame to the STF processing queue."""
        if self._is_closed:
            raise RuntimeError("FaceAnimatorStream is closed")
        
        # Start metrics tracking on first frame (for animation_with_audio mode)
        if self._started_time == 0:
            self._started_time = time.perf_counter()
            self._request_id = str(uuid.uuid4())
        
        self._audio_queue.put_nowait(frame)

    def end_input(self) -> None:
        """Signal that audio input is complete."""
        # if not self._is_closed:
        self._audio_queue.put_nowait(None)  # Use None as a sentinel

    def flush(self) -> None:
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()
        self.end_input()


    async def _send_inference_request(self, audio_data: np.ndarray) -> np.ndarray | None:
        """Send inference request to the inference server."""
        return await self._send_inference_server_request(audio_data)
    
    async def _send_inference_server_request(self, audio_data: np.ndarray) -> np.ndarray | None:
        """Send inference request to inference server with resampled audio."""
        try:
            # Convert int16 -> float32 and resample to 16kHz if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32)
            else:
                audio_float = audio_data
            
            if hasattr(self, '_original_sample_rate') and self._original_sample_rate is not None and self._original_sample_rate != 16000:
                resampled_audio = librosa.resample(
                    audio_float, 
                    orig_sr=self._original_sample_rate, 
                    target_sr=16000
                )
            else:
                resampled_audio = audio_float
            
            # Prepare audio input (1D float32 array)
            audio_input = resampled_audio.astype(np.float32)
            # Audio scalar - using default value from tm_test.py
            # audio_scalar = np.array([1.2], dtype=np.float32)
            
            # Create inference server input objects
            input_audio = httpclient.InferInput("audio_chunk", audio_input.shape, "FP32")
            # input_scalar = httpclient.InferInput("audio_scalar", audio_scalar.shape, "FP32")
            input_audio.set_data_from_numpy(audio_input)
            # input_scalar.set_data_from_numpy(audio_scalar)
            
            # Send inference request
            response = self._face_animator._client.infer(
                self._face_animator._model_name,
                # inputs=[input_audio, input_scalar],
                inputs=[input_audio],
            )
            
            # Get output - shape is (1, num_frames, 52)
            output = response.as_numpy("face_animation")
            return output
            
        except Exception as e:
            logger.error(f"Error during inference server request: {e}", exc_info=True)
            return None


    async def _process_frames(self) -> None:
        """Process audio frames and get blendshape data from the server."""
        if self._output_mode == OutputMode.ANIMATION_WITH_AUDIO:
            await self._process_frames_with_audio()
        else:
            await self._process_frames_animation_only()
    
    async def _process_frames_animation_only(self) -> None:
        """Process frames for animation-only output mode."""
        audio_buffer = np.array([], dtype=np.float32)
        min_samples = int(self._chunk_duration_sec * 16000)
        frames_processed = 0
        animations_generated = 0
        start_time = time.time()
        
        logger.info(f"FaceAnimator processing started (animation-only mode)")
        
        try:
            while True:
                frame = await self._audio_queue.get()
                if frame is None:
                    logger.debug("Input end signal received")
                    break
                
                frames_processed += 1
                
                # Convert frame data to float32
                frame_data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
                if np.max(np.abs(frame_data)) > 1.0:
                    frame_data /= 32768.0
                
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
                    audio_buffer = audio_buffer[min_samples:]
                    
                    animation_output = await self._send_inference_request(audio_to_process)
                    
                    if animation_output is not None and animation_output.size > 0:
                        # Record TTFF on first output
                        if not self._first_output_recorded and self._started_time > 0:
                            self._ttff = time.perf_counter() - self._started_time
                            self._first_output_recorded = True
                            logger.debug(f"First frame generated: TTFF={self._ttff*1000:.0f}ms")
                        
                        # Output is (1, num_frames, 52) - need to flatten
                        flattened_output = animation_output[0]
                        for blendshape_frame in flattened_output:
                            await self._output_queue.put(blendshape_frame)
                            animations_generated += 1
            
            # Process remaining buffer
            if len(audio_buffer) > 0:
                animation_output = await self._send_inference_request(audio_buffer)
                if animation_output is not None and animation_output.size > 0:
                    flattened_output = animation_output[0]
                    for blendshape_frame in flattened_output:
                        await self._output_queue.put(blendshape_frame)
                        animations_generated += 1
        
        except Exception as e:
            logger.error(f"FaceAnimator processing error: {e}", exc_info=True)
        finally:
            duration = time.time() - start_time
            # Emit metrics
            if self._started_time > 0 and self._first_output_recorded:
                total_duration = time.perf_counter() - self._started_time
                metrics = STFMetrics(
                    label=f"{self._face_animator._model_name}_streaming",
                    request_id=self._request_id,
                    timestamp=time.time(),
                    duration=total_duration,
                    ttff=self._ttff,
                    frames_generated=animations_generated,
                    audio_duration=duration,
                )
                self._face_animator._emit_metrics(metrics)
                logger.debug(f"Metrics emitted: TTFF={self._ttff*1000:.0f}ms")

            await self._output_queue.put(None)
    
    async def _process_frames_with_audio(self) -> None:
        """Process frames for animation+audio output mode."""
        audio_buffer = np.array([], dtype=np.int16)  # Keep original int16 type
        frames_processed = 0
        animations_generated = 0
        start_time = time.time()
        self._original_sample_rate = None
        
        logger.info(f"FaceAnimator processing started (animation+audio mode)")
        
        try:
            while True:
                frame = await self._audio_queue.get()
                if frame is None:
                    break
                
                frames_processed += 1
                
                # Store sample rate from first frame
                if self._original_sample_rate is None:
                    self._original_sample_rate = frame.sample_rate
                
                # Convert frame data to int16 (keep original type)
                frame_data = np.frombuffer(frame.data, dtype=np.int16)
                
                # Add to buffer (keep original sample rate)
                audio_buffer = np.concatenate((audio_buffer, frame_data))
                
                # Process audio in chunks (based on original sample rate)
                min_samples = int(self._chunk_duration_sec * self._original_sample_rate)
                
                while len(audio_buffer) >= min_samples:
                    audio_to_process = audio_buffer[:min_samples]
                    audio_buffer = audio_buffer[min_samples:]
                    
                    # Send request to server (conversion and resampling done internally)
                    animation_output = await self._send_inference_request(audio_to_process)
                    
                    if animation_output is not None and animation_output.size > 0:
                        # Record TTFF on first output
                        if not self._first_output_recorded and self._started_time > 0:
                            self._ttff = time.perf_counter() - self._started_time
                            self._first_output_recorded = True
                            logger.debug(f"First frame generated: TTFF={self._ttff*1000:.0f}ms")
                        
                        # Output shape is (1, num_frames, 52) - need to flatten first dimension
                        flattened_output = animation_output[0]
                        num_frames = flattened_output.shape[0]
                        
                        # Evenly distribute audio samples to animation frames
                        total_samples = len(audio_to_process)
                        
                        for i, blendshape_frame in enumerate(flattened_output):
                            # Calculate sample range for each frame
                            start_sample = (i * total_samples) // num_frames
                            end_sample = ((i + 1) * total_samples) // num_frames if i < num_frames - 1 else total_samples
                            
                            # Extract audio chunk (pass as int16 numpy array)
                            audio_chunk_samples = audio_to_process[start_sample:end_sample]
                            
                            await self._output_queue.put((
                                blendshape_frame,
                                audio_chunk_samples,
                                self._original_sample_rate or 16000,
                                frame.num_channels
                            ))
                            animations_generated += 1
            
            # Process remaining buffer
            if len(audio_buffer) > 0:
                animation_output = await self._send_inference_request(audio_buffer)
                if animation_output is not None and animation_output.size > 0:
                    flattened_output = animation_output[0]
                    num_frames = flattened_output.shape[0]
                    total_samples = len(audio_buffer)
                    
                    for i, blendshape_frame in enumerate(flattened_output):
                        start_sample = (i * total_samples) // num_frames
                        end_sample = ((i + 1) * total_samples) // num_frames if i < num_frames - 1 else total_samples
                        audio_chunk_samples = audio_buffer[start_sample:end_sample]
                        
                        await self._output_queue.put((
                            blendshape_frame,
                            audio_chunk_samples,
                            self._original_sample_rate or 16000,
                            1  # Default to mono
                        ))
                        animations_generated += 1
        
        except Exception as e:
            logger.error(f"FaceAnimator processing error: {e}", exc_info=True)
        finally:
            duration = time.time() - start_time
            
            # Emit metrics
            if self._started_time > 0 and self._first_output_recorded:
                total_duration = time.perf_counter() - self._started_time
                metrics = STFMetrics(
                    label=f"{self._face_animator._model_name}_streaming",
                    request_id=self._request_id,
                    timestamp=time.time(),
                    duration=total_duration,
                    ttff=self._ttff,
                    frames_generated=animations_generated,
                    audio_duration=duration,
                )
                self._face_animator._emit_metrics(metrics)
                logger.debug(f"Metrics emitted: TTFF={self._ttff*1000:.0f}ms")
            
            await self._output_queue.put(None)

    def __aiter__(self) -> AsyncIterator[AnimationData]:
        return self

    async def __anext__(self) -> AnimationData:
        """Return animation data."""
        if self._task is None:
            logger.debug("FaceAnimatorStream processing task started")
            self._task = asyncio.create_task(self._process_frames())
            self._last_frame_time = asyncio.get_event_loop().time()

        output = await self._output_queue.get()
        if output is None:
            logger.debug("FaceAnimatorStream ended")
            # Ensure task is finished before raising StopAsyncIteration
            if self._task and not self._task.done():
                try:
                    await self._task
                except Exception:
                    logger.error("Error during final task wait in FaceAnimatorStream.__anext__", exc_info=True)
            raise StopAsyncIteration

        # Create timestamp
        timestamp_us = int(asyncio.get_event_loop().time() * 1_000_000)
        segment_id = str(uuid.uuid4())

        # Handle output based on mode
        if self._output_mode == OutputMode.ANIMATION_ONLY:
            # Output is just the blendshape frame
            animation_data = AnimationData.from_numpy(
                output, timestamp_us=timestamp_us, segment_id=segment_id
            )
        else:
            # Output is (blendshape_frame, audio_samples, sample_rate, num_channels)
            blendshape_frame, audio_samples, sample_rate, num_channels = output
            animation_data = AnimationData.from_pair(
                animation_arr=blendshape_frame,
                audio_samples=audio_samples,
                sample_rate=sample_rate,
                num_channels=num_channels,
                timestamp_us=timestamp_us,
                segment_id=segment_id
            )

        return animation_data

    async def aclose(self) -> None:
        """Close the stream."""
        if not self._is_closed:
            logger.debug("FaceAnimatorStream aclose called")
            self.flush()
            self._is_closed = True
            if self._task is not None:
                await aio.cancel_and_wait(self._task)

    async def __aenter__(self) -> "FaceAnimatorStream":
        """Enter asynchronous context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit asynchronous context and clean up resources."""
        await self.aclose()


class FaceAnimator(STF):
    """Unified Speech-To-Face implementation supporting multiple backends and output modes."""
    
    def __init__(
        self,
        *,
        # Server configuration
        server_url: str = "localhost:8401",
        model_name: str = "ensemble_face",
        
        # Common parameters
        frame_rate: int = 60,
        sample_rate: int = 16000,
        num_features: int = 52,
        chunk_duration_sec: float = 0.5,
        
        # Output configuration
        output_mode: OutputMode | str = OutputMode.ANIMATION_WITH_AUDIO,
    ) -> None:
        super().__init__()  # 부모 클래스 초기화 호출
        
        # Server configuration
        self._server_url = server_url
        self._model_name = model_name
        
        # Common parameters
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._num_features = num_features
        self._chunk_duration_sec = chunk_duration_sec
        
        # Output mode
        if isinstance(output_mode, str):
            output_mode = OutputMode(output_mode)
        self._output_mode = output_mode
        
        # Create inference server HTTP client
        self._client = httpclient.InferenceServerClient(url=server_url)
        logger.info(f"Initialized FaceAnimator with inference server: {server_url}, model: {model_name}")
        
        # Keep track of active streams
        self._streams = weakref.WeakSet[FaceAnimatorStream]()
    
    def _emit_metrics(self, metrics: STFMetrics) -> None:
        """Emit STF metrics to event listeners."""
        self.emit("metrics_collected", metrics)
    
    async def aclose(self) -> None:
        """Close the FaceAnimator and associated streams."""
        # Close all active streams
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
    
    def stream(self) -> FaceAnimatorStream:
        """Create a new FaceAnimator stream."""
        stream = FaceAnimatorStream(
            self, 
            chunk_duration_sec=self._chunk_duration_sec,
            output_mode=self._output_mode
        )
        self._streams.add(stream)
        return stream
    
    async def synthesize(
        self, audio_stream: AsyncIterable[rtc.AudioFrame]
    ) -> AsyncIterable[AnimationData]:
        """Generate face animation data from an audio stream."""
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
        self, audio_stream: AsyncIterable[rtc.AudioFrame], stream: FaceAnimatorStream
    ) -> None:
        """Forwards audio frames from an iterable to the FaceAnimator stream."""
        try:
            async for frame in audio_stream:
                stream.push_frame(frame)
        except Exception as e:
            logger.error(f"Error forwarding audio to FaceAnimator stream: {e}", exc_info=True)
        finally:
            # Signal end of audio input
            stream.end_input()