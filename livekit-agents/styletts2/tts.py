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
    emotion: str


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
        emotion: str = "Neutral",
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
            emotion=emotion,
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
        emotion: Optional[str] = None,
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
        if emotion is not None:
            self._opts.emotion = emotion

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
            "format": "pcm",
            "sample_rate": self._opts.sample_rate,
            "emotion": self._opts.emotion,
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
                
                # Create a single AudioFrame with the raw audio data
                frame = rtc.AudioFrame(
                    data=audio_data,
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    samples_per_channel=len(audio_data) // 2  # int16 = 2 bytes
                )
                
                # Yield the synthesized audio
                yield tts.SynthesizedAudio(
                    request_id=request_id,
                    segment_id=utils.shortuuid(),
                    frame=frame,
                )
                
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