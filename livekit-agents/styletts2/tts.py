from __future__ import annotations

import asyncio
import os
import numpy as np
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import aiohttp
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from scipy import signal

from .log import logger

@dataclass
class _TTSOptions:
    base_url: str
    sample_rate: int
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    """StyleTTS2 TTS implementation for LiveKit agents"""
    
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8014",
        sample_rate: int = 24000,
        word_tokenizer: Optional[tokenize.WordTokenizer] = None,
        http_session: Optional[aiohttp.ClientSession] = None,
        num_channels: int = 1,  # 기본값을 1로 변경 (모노)
    ) -> None:
        """
        Create a new instance of StyleTTS2 TTS.

        Args:
            base_url (str): Base URL for the StyleTTS2 server. Defaults to "http://localhost:8014".
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text.
            http_session (aiohttp.ClientSession): Optional HTTP session for API requests.
            num_channels (int): Number of audio channels. Defaults to 1 (mono).
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        if word_tokenizer is None:
            word_tokenizer = tokenize.basic.WordTokenizer(
                ignore_punctuation=False
            )

        self._opts = _TTSOptions(
            base_url=base_url,
            sample_rate=sample_rate,
            word_tokenizer=word_tokenizer,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        base_url: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        """
        Update the TTS options.

        Args:
            base_url (str): Base URL for the StyleTTS2 server.
            sample_rate (int): Sample rate of audio.
        """
        if base_url is not None:
            self._opts.base_url = base_url
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate

    async def synthesize(
        self,
        text: str,
        *,
        conn_options: Optional[APIConnectOptions] = None,
    ) -> AsyncGenerator[tts.SynthesizedAudio, None]:
        """
        Synthesize text to audio.

        Args:
            text (str): Text to synthesize.
            conn_options (APIConnectOptions): Connection options.
        """
        request_id = utils.shortuuid()
        text = text.replace('"', '').strip()
        
        if len(text) <= 1:
            logger.debug(f'Skipping TTS: [{text}]')
            return
            
        logger.debug(f"Generating TTS: [{text}]")
        
        url = self._opts.base_url + "/tts"
        
        payload = {
            "text": text,
        }
        
        try:
            # Ensure session is initialized
            self._session = self._ensure_session()
            
            async with self._session.post(url, json=payload) as r:
                if r.status != 200:
                    text = await r.text()
                    logger.error(f"StyleTTS2 error getting audio (status: {r.status}, error: {text})")
                    raise APIStatusError(
                        message=f"Error getting audio from StyleTTS2: {text}",
                        status_code=r.status,
                        request_id=request_id,
                        body=None,
                    )
                 
                # Read all audio data at once
                audio_data = await r.read()
                
                # Convert audio data to numpy array (16-bit PCM to float32)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                logger.info(f"[DEBUG] Initial audio data: shape={audio_np.shape}, dtype={audio_np.dtype}, samples={len(audio_np)}")
                
                # Resample to match requested sample_rate if needed
                if self._opts.sample_rate != 24000:
                    # Convert to float32 for resampling
                    audio_f32 = audio_np.astype(np.float32) / 32768.0
                    logger.info(f"[DEBUG] Converted to float32: shape={audio_f32.shape}, dtype={audio_f32.dtype}")
                    
                    # Calculate new length
                    new_length = int(len(audio_f32) * self._opts.sample_rate / 24000)
                    logger.info(f"[DEBUG] Resampling from {len(audio_f32)} to {new_length} samples")
                    
                    # Apply scipy resampling
                    audio_resampled = signal.resample(audio_f32, new_length)
                    logger.info(f"[DEBUG] After resampling: shape={audio_resampled.shape}, dtype={audio_resampled.dtype}")
                    
                    # Convert back to int16
                    audio_np = (audio_resampled * 32767).astype(np.int16)
                    logger.info(f"[DEBUG] Final audio data: shape={audio_np.shape}, dtype={audio_np.dtype}, samples={len(audio_np)}")
                    logger.info(f"Resampled audio from 24000Hz to {self._opts.sample_rate}Hz")
                
                # 오디오 데이터를 100ms 청크로 나누기
                samples_per_frame = self._opts.sample_rate // 10  # 100ms at sample_rate
                total_100ms_chunks = (len(audio_np) + samples_per_frame - 1) // samples_per_frame
                total_samples = total_100ms_chunks * samples_per_frame
                
                logger.info(f"Audio length: {len(audio_np)} samples, padding to {total_samples} samples")
                logger.info(f"Total chunks: {total_100ms_chunks} chunks of 100ms")
                
                # 각 청크별로 오디오 프레임 생성
                bytes_per_sample = 2  # int16 = 2 bytes
                for chunk_idx in range(total_100ms_chunks):
                    chunk_start = chunk_idx * samples_per_frame
                    chunk_end = (chunk_idx + 1) * samples_per_frame
                    
                    # 배열 범위를 벗어나지 않도록 체크
                    if chunk_start >= len(audio_np):
                        break
                        
                    # 현재 청크의 데이터 추출
                    chunk_data = audio_np[chunk_start:chunk_end].tobytes()
                    
                    # 청크 데이터 길이가 충분한지 확인
                    expected_size = samples_per_frame * bytes_per_sample
                    if len(chunk_data) < expected_size:
                        # 부족한 부분을 0으로 패딩
                        padding_needed = expected_size - len(chunk_data)
                        chunk_data = chunk_data + bytes(padding_needed)
                        logger.info(f"Padded chunk {chunk_idx+1} with {padding_needed} zero bytes")
                    
                    # 디버깅 정보
                    logger.info(f"Creating frame {chunk_idx+1}/{total_100ms_chunks}: data size={len(chunk_data)} bytes")
                    
                    # AudioFrame 직접 생성
                    frame = rtc.AudioFrame(
                        data=chunk_data,
                        sample_rate=self._opts.sample_rate,
                        num_channels=1,  # 모노 채널
                        samples_per_channel=samples_per_frame,
                    )
                    
                    logger.info(f"Generated audio frame {chunk_idx+1}: sample_rate={frame.sample_rate}, num_channels={frame.num_channels}, samples={len(frame.data) // (2 * frame.num_channels)}")
                    
                    # 프레임 반환
                    yield tts.SynthesizedAudio(
                        request_id=request_id,
                        segment_id=utils.shortuuid(),
                        frame=frame,
                    )
                
                # 디버깅을 위한 로그 추가
                logger.info(f"[DEBUG] Audio processing completed with sample_rate={self._opts.sample_rate}, num_channels=1")
                logger.info(f"[DEBUG] Audio data: shape={audio_np.shape}, dtype={audio_np.dtype}, samples={len(audio_np)}")
                
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    async def aclose(self) -> None:
        await super().aclose() 