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
from livekit.plugins import deepgram, openai, silero

# STF 모듈 임포트 및 기본 URL 정의
from livekit.agents.stf import FaceAnimatorSTF
from livekit.agents.voice.agent import Agent
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.room_io.room_io import RoomInputOptions, RoomOutputOptions, RoomIO

load_dotenv()  # .env 파일에서 환경 변수 로드
logger = logging.getLogger("face-animation-agent")

def prewarm(proc: JobProcess):
    # VAD 모델 로드
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # 기본 시스템 메시지 설정
    initial_ctx = llm.ChatContext()
    initial_ctx.add_message(
        role="system",
        content=(
            "당신은 LiveKit에서 만든 음성 도우미입니다. 사용자와의 인터페이스는 음성으로 이루어집니다. "
            "짧고 간결한 응답을 사용하고, 발음할 수 없는 문장 부호 사용을 피하세요."
        ),
    )

    logger.info(f"{ctx.room.name} 방에 연결합니다")
    # 오디오만 구독 (STT 용)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 첫 번째 참가자가 연결될 때까지 대기
    participant = await ctx.wait_for_participant()
    logger.info(f"{participant.identity} 참가자를 위한 음성-얼굴 에이전트 시작")

    # 에이전트 생성
    agent = Agent(
        instructions=(
            "당신은 LiveKit에서 만든 음성 도우미입니다. 사용자와의 인터페이스는 음성으로 이루어집니다. "
            "짧고 간결한 응답을 사용하고, 발음할 수 없는 문장 부호 사용을 피하세요."
        ),
        chat_ctx=initial_ctx,
        llm=openai.LLM(model="gpt-4o-mini"),  # 사용할 LLM 모델
    )

    # AgentSession 생성 (STF 클라이언트 포함)
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(model="gpt-4o-mini-transcribe"),  # OpenAI Whisper STT 모델 사용
        tts=openai.TTS(voice="alloy"),  # 음성 기본 설정
        stf=FaceAnimatorSTF(),  # STF 클라이언트 설정
    )

    # RoomIO 옵션 설정 (애니메이션 데이터 출력 활성화)
    room_output_options = RoomOutputOptions(
        video_enabled=False,          # 비디오 출력 비활성화
        audio_enabled=True,           # 오디오 출력
        transcription_enabled=True,   # 텍스트 전사 출력
        animation_enabled=True,       # 애니메이션 데이터 출력 활성화
    )

    # RoomIO 생성 및 참가자 설정
    room_io = RoomIO(
        session,
        ctx.room,
        participant=participant,  # 여기서 직접 참가자 설정
        output_options=room_output_options
    )

    logger.info(f"애니메이션 데이터 스트리밍을 활성화했습니다. 대상: {participant.identity}")

    # 세션 시작 - room_io 사용
    await room_io.start()
    await session.start(agent)

    # 사용량 수집기 설정
    usage_collector = metrics.UsageCollector()

    # 초기화 완료까지 잠시 대기
    await asyncio.sleep(1)

    # 인사말 출력
    await session.say("안녕하세요! 애니메이션 얼굴을 가진 AI 도우미입니다. 무엇을 도와드릴까요?")

    # 세션이 닫힐 때까지 대기
    close_event = asyncio.Event()
    session.on("close", lambda _: close_event.set())
    await close_event.wait()

    # 종료 시 리소스 정리
    await room_io.aclose()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    ) 