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
import aiohttp # Added aiohttp
import time
import uuid

from ..log import logger
from ..utils import aio, http_context
from .animation_data import AnimationData   

# Default URL for the STF server
DEFAULT_STF_SERVER_URL = "http://localhost:8015/stf"


class STF(Protocol):
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


class STFStream:
    """Speech-to-Face stream class - Processes audio and gets blendshapes from a server."""

    def __init__(self, stf: "FaceAnimatorSTF", chunk_duration_sec: float) -> None:
        self._stf = stf  # Now expects FaceAnimatorSTF instance
        self._audio_queue = asyncio.Queue[Optional[rtc.AudioFrame]]()
        self._is_closed = False
        self._task: Optional[asyncio.Task] = None
        self._blendshape_frames = asyncio.Queue[Optional[np.ndarray]]()  # Queue for numpy arrays
        self._last_frame_time = 0.0
        self._chunk_duration_sec = chunk_duration_sec

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        """Add audio frame to the STF processing queue."""
        if self._is_closed:
            raise RuntimeError("STFStream is closed")
        self._audio_queue.put_nowait(frame)

    def end_input(self) -> None:
        """Signal that audio input is complete."""
        # if not self._is_closed:
        self._audio_queue.put_nowait(None)  # Use None as a sentinel

    def flush(self) -> None:
        while not self._audio_queue.empty():
            self._audio_queue.get_nowait()
        self.end_input()

    async def _preprocess_audio_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocesses a chunk of audio data (normalization, reshaping)."""
        # Already resampled to 16kHz in the loop
        if np.any(audio_data):
            mean = np.mean(audio_data)
            variance = np.var(audio_data)
            audio_feats = (audio_data - mean) / np.sqrt(variance + 1e-7)
        else:
            logger.warning("Audio chunk appears to be silent, skipping normalization.")
            audio_feats = audio_data
        # Reshape for the server (1, num_samples)
        return audio_feats.reshape(1, -1)

    async def _send_inference_request(self, audio_features: np.ndarray) -> Optional[np.ndarray]:
        """Sends preprocessed audio features to the STF server."""
        session = self._stf._ensure_session()
        payload = {
            "audio_features": audio_features.tolist(),
            "frame_rate": self._stf._frame_rate,
        }
        request_start_time = time.time()
        try:
            # Added timeout
            async with session.post(self._stf._server_url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response_time = time.time() - request_start_time
                # logger.debug(
                #     f"STF server request sent (Audio duration: {audio_features.shape[1]/16000:.2f}s). "
                #     f"Response: {response.status} ({response_time:.2f}s)"
                # )
                response.raise_for_status()
                result = await response.json()
                blendshapes_list = result.get("blendshapes")
                if blendshapes_list is not None:
                    # Convert list of lists back to numpy array
                    blendshapes_array = np.array(blendshapes_list, dtype=np.float32)
                    # logger.debug(f"Received {blendshapes_array.shape[0]} blendshape frames from server.")
                    return blendshapes_array
                else:
                    logger.warning("STF server response did not contain 'blendshapes'.")
                    return None
        except asyncio.TimeoutError:
             logger.error(f"Timeout connecting to STF server at {self._stf._server_url}")
             return None
        except aiohttp.ClientResponseError as e:
            logger.error(f"STF server returned error status {e.status}: {e.message}")
            try:
                # Attempt to get error detail from server
                error_detail = await response.json()
                logger.error(f"Server error detail: {error_detail}")
            except Exception:
                 pass # Ignore if getting JSON detail fails
            return None
        except aiohttp.ClientConnectorError as e:
             logger.error(f"Could not connect to STF server at {self._stf._server_url}: {e}")
             return None
        except Exception as e:
            logger.error(f"Error during STF inference request: {e}", exc_info=True)
            return None


    async def _process_frames(self) -> None:
        """Processes audio frames and gets blendshape data from the server."""
        audio_buffer = np.array([], dtype=np.float32)
        # Use chunk duration consistent with test client or configurable? Let's use 1 second for now.
        min_samples = int(self._chunk_duration_sec * 16000)  # Process in chunks of 1 second
        frames_processed = 0
        animations_generated = 0
        start_time = time.time()

        logger.info(f"STF 프레임 처리 시작 (서버: {self._stf._server_url}, 청크 크기: {self._chunk_duration_sec}초)")

        try:
            while True:
                frame = await self._audio_queue.get()

                if frame is None:
                    logger.debug("STF 입력 종료 신호 수신")
                    break

                frames_processed += 1

                # Convert frame data to float32
                frame_data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
                # Check max absolute value
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
                    # Keep remaining audio
                    audio_buffer = audio_buffer[min_samples:]

                    # logger.debug(f"STF 처리 청크 준비 (길이: {len(audio_to_process)/16000.0:.2f}초)")
                    # Preprocess the chunk
                    audio_feats = await self._preprocess_audio_chunk(audio_to_process)

                    if audio_feats.size > 0:
                        # Send request to server
                        animation_output = await self._send_inference_request(audio_feats)

                        if animation_output is not None and animation_output.size > 0:
                            num_frames = animation_output.shape[0]
                            # logger.debug(f"서버로부터 {num_frames}개 블렌드쉐입 프레임 수신")
                            # Put each frame (numpy array) into the queue
                            for blendshape_frame in animation_output:
                                await self._blendshape_frames.put(blendshape_frame)
                                animations_generated += 1

                            # Periodic logging
                            # if animations_generated % (self._stf._frame_rate * 5) == 0:
                            #     elapsed = time.time() - start_time
                            #     fps = animations_generated / elapsed if elapsed > 0 else 0
                            #     logger.debug(f"애니메이션 생성 진행: {animations_generated}개 프레임, 실시간 FPS 추정: {fps:.1f}")
                    else:
                         logger.warning("Skipping inference request for empty preprocessed chunk.")


            # Process any remaining audio in the buffer after input ends
            if len(audio_buffer) > 0:
                buffer_duration = len(audio_buffer) / 16000.0
                # logger.debug(f"남은 오디오 처리: {buffer_duration:.2f}초 ({len(audio_buffer)} 샘플)")

                audio_feats = await self._preprocess_audio_chunk(audio_buffer)

                if audio_feats.size > 0:
                    animation_output = await self._send_inference_request(audio_feats)
                    if animation_output is not None and animation_output.size > 0:
                        num_frames = animation_output.shape[0]
                        # logger.debug(f"최종 STF 추론: 서버로부터 {num_frames}개 프레임 수신")
                        for blendshape_frame in animation_output:
                            await self._blendshape_frames.put(blendshape_frame)
                            animations_generated += 1
                else:
                    logger.warning("Skipping inference request for final empty preprocessed chunk.")


        except Exception as e:
            logger.error(f"STF 프레임 처리 중 오류 발생: {e}", exc_info=True)
        finally:
            duration = time.time() - start_time
            logger.info(
                f"STF 프레임 처리 완료: {frames_processed}개 오디오 프레임 처리, "
                f"{animations_generated}개 애니메이션 생성, 총 소요 시간: {duration:.2f}초"
            )
            # Use None as the end marker
            await self._blendshape_frames.put(None)

    def __aiter__(self) -> AsyncIterator[AnimationData]:
        return self

    async def __anext__(self) -> AnimationData:
        """Return animation data."""
        if self._task is None:
            logger.debug("STF 프레임 처리 태스크 시작")
            self._task = asyncio.create_task(self._process_frames())
            self._last_frame_time = asyncio.get_event_loop().time()

        blendshape_frame = await self._blendshape_frames.get()
        if blendshape_frame is None:
            logger.debug("STF 스트림 종료")
            # Ensure task is finished before raising StopAsyncIteration
            if self._task and not self._task.done():
                try:
                    await self._task
                except Exception:
                    logger.error("Error during final task wait in STFStream.__anext__", exc_info=True)
            raise StopAsyncIteration

        # Create timestamp
        timestamp_us = int(asyncio.get_event_loop().time() * 1_000_000)
        segment_id = str(uuid.uuid4())

        # Create AnimationData object directly from the blendshape numpy array
        animation_data = AnimationData.from_numpy(
            blendshape_frame, timestamp_us=timestamp_us, segment_id=segment_id
        )

        # Frame rate logging (optional)
        # now = asyncio.get_event_loop().time()
        # frame_time = now - self._last_frame_time
        # self._last_frame_time = now
        # logger.debug(f"Generated animation data frame. Features: {animation_data.num_features}, Frame time: {frame_time*1000:.1f}ms")

        return animation_data

    async def aclose(self) -> None:
        """Close the stream."""
        if not self._is_closed:
            # Ensure the processing loop finishes
            # self.end_input()
            logger.debug("STF 스트림 aclose 호출")
            self.flush()
            self._is_closed = True
            await aio.cancel_and_wait(self._task)



            # if self._task is not None:
            #     try:
            #         # Wait for task to finish, handling cancellation
            #         await asyncio.wait_for(self._task, timeout=5.0)
            #     except asyncio.CancelledError:
            #          logger.debug("STF processing task cancelled during close.")
            #     except asyncio.TimeoutError:
            #          logger.warning("Timeout waiting for STF processing task during close. Cancelling.")
            #          self._task.cancel()
            #          try:
            #              await self._task # Allow cancellation to propagate
            #          except asyncio.CancelledError:
            #              pass
            #     except Exception:
            #          logger.error("Error during STFStream task cleanup", exc_info=True)
            # Clear queues? Maybe not necessary if task completion handles it.


    async def __aenter__(self) -> "STFStream":
        """Enter asynchronous context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit asynchronous context and clean up resources."""
        await self.aclose()


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



class FaceAnimatorSTF(STF):
    """Implementation using a remote STF server."""

    def __init__(
        self,
        *,
        server_url: str = DEFAULT_STF_SERVER_URL,
        frame_rate: int = 60, # Target animation frame rate sent to server
        # These parameters are now primarily informational or defaults
        sample_rate: int = 16000, # Expected input sample rate (client enforces this)
        num_features: int = 52, # Expected number of blendshape features from server
        chunk_duration_sec: float = 1.0, # Client-side chunking duration (in STFStream)
        http_session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._server_url = server_url
        self._sample_rate = sample_rate
        self._frame_rate = frame_rate
        self._num_features = num_features
        # Used by STFStream chunking logic
        self._chunk_duration_sec = chunk_duration_sec

        # Remove local ONNX model attributes
        # self._audio_enc_session = None
        # self._diffusion_session = None

        # Add HTTP session management
        self._session = http_session
        # Keep track of active streams
        self._streams = weakref.WeakSet[STFStream]()

        logger.info(f"Initialized FaceAnimatorSTF client for server: {server_url}")

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp ClientSession."""
        if not self._session:
            # Use shared session from http_context
            self._session = http_context.http_session()
        return self._session

    async def aclose(self) -> None:
        """Close the FaceAnimatorSTF client and associated streams."""
        logger.debug("Closing FaceAnimatorSTF client and streams.")
        # Close all active streams associated with this instance
        # Iterate over a copy
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        # Do not close the shared session here if obtained from http_context
        # await super().aclose() # If STF inherited from a closable base

    # Removed local inference methods:
    # _load_sessions
    # _infer_onnx
    # _infer_face_diffusion_async
    # _infer_face_diffusion
    # _create_video_frame (was already deprecated)

    def stream(self) -> STFStream:
        """Create a new STF stream."""
        stream = STFStream(self, chunk_duration_sec=self._chunk_duration_sec)
        # Track the created stream
        self._streams.add(stream)
        return stream

    async def synthesize(
        self, audio_stream: AsyncIterable[rtc.AudioFrame]
    ) -> AsyncIterable[AnimationData]:
        """
        Generate face animation data from an audio stream using the server.
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
        self, audio_stream: AsyncIterable[rtc.AudioFrame], stream: STFStream
    ) -> None:
        """Forwards audio frames from an iterable to the STF stream."""
        try:
            async for frame in audio_stream:
                stream.push_frame(frame)
        except Exception as e:
             logger.error(f"Error forwarding audio to STF stream: {e}", exc_info=True)
        finally:
            # Signal end of audio input
            stream.end_input()


class FaceAnimatorSTFTriton(STF):
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

