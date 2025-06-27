import asyncio
import logging
import os

# Set higher logging level for Numba before other configurations
logging.getLogger('numba').setLevel(logging.WARNING)

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
    utils,
)
from livekit.plugins import deepgram, openai, silero, elevenlabs

# STF 모듈 임포트 및 기본 URL 정의
from livekit.agents.stf import FaceAnimatorSTFTriton
from livekit.agents.voice.agent import Agent
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.room_io.room_io import RoomInputOptions, RoomOutputOptions

load_dotenv()  # .env 파일에서 환경 변수 로드
logger = logging.getLogger("face-animation-agent")


class FaceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "당신은 LiveKit에서 만든 음성 도우미입니다. 사용자와의 인터페이스는 음성으로 이루어집니다. "
                "짧고 간결한 응답을 사용하고, 발음할 수 없는 문장 부호 사용을 피하세요."
            ),
            stf=FaceAnimatorSTFTriton(chunk_duration_sec=1.0),  
        )

    async def on_enter(self): 
        logger.info("FaceAgent on_enter")
        self.session.generate_reply(instructions="사용자에게 흥미롭고 아주 긴 1문장 건네는 것으로 시작해.") 
        # self.session.generate_reply(instructions="사용자에게 '안녕하세요! 반갑습니다.' 라고 해.'") 


def prewarm(proc: JobProcess):
    # VAD 모델 로드
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info(f"{ctx.room.name} 방에 연결합니다")
    # 오디오만 구독 (STT 용)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 첫 번째 참가자가 연결될 때까지 대기
    participant = await ctx.wait_for_participant()
    logger.info(f"{participant.identity} 참가자를 위한 음성-얼굴 에이전트 시작")

    # AgentSession 생성 (STF 클라이언트 포함)
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # stt=openai.STT(model="gpt-4o-mini-transcribe"),  # OpenAI Whisper STT 모델 사용
        stt=deepgram.STT(model="nova-2-general", language="ko"),
        llm=openai.LLM(model="gpt-4.1-nano"),
        # llm=openai.realtime.RealtimeModel(model="gpt-4o-realtime-preview-2025-06-03"),
        # tts=openai.TTS(model="gpt-4o-mini-tts", voice="alloy"),  # 음성 기본 설정 
        tts=elevenlabs.TTS(
                voice_id="SHi5MVTovxhdsNpOHkyG",
                model="eleven_turbo_v2_5",
                voice_settings=elevenlabs.VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.0,
                    speed=1.0,
                ),
                encoding="mp3_44100_32",
            ),
    )

    room_input_options = RoomInputOptions(
        audio_enabled=True,
        video_enabled=False,
        text_enabled=False,
        # text_enabled=False,
        participant_identity=participant.identity,
    )
    # RoomIO 옵션 설정 (애니메이션 데이터 출력 활성화)
    room_output_options = RoomOutputOptions(
        audio_enabled=False,          # 오디오 출력 비활성화 (AnimationData에 포함됨)
        transcription_enabled=True,   # 텍스트 전사 출력
        animation_enabled=True,       # 애니메이션 데이터 출력 활성화
        sync_transcription=False,
    )

    logger.info(f"애니메이션 데이터 스트리밍을 활성화했습니다. 대상: {participant.identity}")

    # session.output.audio = DataStreamAudioOutput(
    #         room=ctx.room,
    #         destination_identity=participant.identity,
    #     )
    # 세션 시작
    await session.start(
        agent=FaceAgent(),
        room=ctx.room,
        room_input_options=room_input_options,
        room_output_options=room_output_options
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    ) 