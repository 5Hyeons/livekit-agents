import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any

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
    RunContext,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, silero, elevenlabs

# STF 모듈 임포트 및 기본 URL 정의
from livekit.agents.stf import FaceAnimatorSTFTriton
from livekit.agents.voice.agent import Agent
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.room_io.room_io import RoomInputOptions, RoomOutputOptions

# 사용자 데이터베이스 임포트
from user_database import UserDatabase, UserData, ChatMessage

load_dotenv()  # .env 파일에서 환경 변수 로드
logger = logging.getLogger("face-animation-agent")


class FaceAgent(Agent):
    def __init__(self, user_data: UserData, db: UserDatabase):
        self.user_data = user_data
        self.db = db
        
        # 이전 대화 컨텍스트 가져오기
        context = self.db.get_recent_context(user_data.participant_id, message_count=20)
        
        # 기본 지시사항에 사용자 정보 포함
        base_instructions = (
            "당신은 LiveKit에서 만든 음성 도우미입니다. 사용자와의 인터페이스는 음성으로 이루어집니다. "
            "짧고 간결한 응답을 사용하고, 발음할 수 없는 문장 부호 사용을 피하세요."
        )
        
        # 사용자 이름이 있으면 추가
        if user_data.display_name:
            base_instructions += f"\n\n사용자의 이름은 '{user_data.display_name}'입니다."
        
        # 이전 대화 컨텍스트가 있으면 추가
        if context:
            base_instructions += f"\n\n이전 대화 내용:\n{context}"
            
        super().__init__(
            instructions=base_instructions,
            stt=deepgram.STT(model="nova-2-general", language="ko"),
            llm=openai.LLM(model="gpt-4o"),
            tts=elevenlabs.TTS(
                    voice_id="tZarJVdIxWQ9lIXIV9qg",
                    model="eleven_turbo_v2_5",
                    voice_settings=elevenlabs.VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.75,
                        style=0.0,
                        speed=1.0,
                    ),
                    encoding="mp3_44100_32",
                ),
            stf=FaceAnimatorSTFTriton(chunk_duration_sec=1.0),
        )

    async def on_enter(self): 
        logger.info(f"FaceAgent on_enter for user: {self.user_data.participant_id}")
        
        if self.user_data.display_name:
            # 이미 이름을 아는 경우
            greeting = f"{self.user_data.display_name}님, 다시 만나서 반가워요! 무엇을 도와드릴까요?"
            self.session.generate_reply(instructions=f"'{greeting}'라고 인사하세요.")
        else:
            # 처음 만나는 경우
            self.session.generate_reply(instructions="사용자에게 간단히 인사를 하고 이름을 물어보는 것으로 시작하세요.")
    
    @function_tool
    async def save_user_name(self, name: str):
        """
        사용자의 이름을 저장합니다. 사용자가 자신의 이름을 알려줄 때 이 함수를 호출하세요.
        이 함수는 한 번만 호출해야 합니다.
        
        Args:
            name: 사용자의 이름
        """
        
        # 이미 이름이 저장되어 있다면 중복 호출 방지
        if self.user_data.display_name and self.user_data.display_name == name:
            logger.info(f"이름이 이미 저장됨: {name}")
            return
        
        self.db.update_user_name(self.user_data.participant_id, name)
        self.user_data.display_name = name
        logger.info(f"사용자 이름 저장: {self.user_data.participant_id} -> {name}")
        result = f"네, {name}님! 이름을 기억했습니다."
        logger.info(f"🔧 [TOOL DEBUG] save_user_name 결과 반환: {result}")
        return result

def prewarm(proc: JobProcess):
    # VAD 모델 로드
    proc.userdata["vad"] = silero.VAD.load()
    # 데이터베이스 초기화
    proc.userdata["db"] = UserDatabase()

async def entrypoint(ctx: JobContext):
    logger.info(f"{ctx.room.name} 방에 연결합니다")
    # 오디오만 구독 (STT 용)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 첫 번째 참가자가 연결될 때까지 대기
    participant = await ctx.wait_for_participant()
    logger.info(f"{participant.identity} 참가자를 위한 음성-얼굴 에이전트 시작")
    
    # 데이터베이스에서 사용자 정보 가져오기 또는 생성
    db: UserDatabase = ctx.proc.userdata["db"]
    user_data = db.get_or_create_user(participant.identity)

    # AgentSession 생성 (STF 클라이언트 포함)
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # stt=openai.STT(model="gpt-4o-mini-transcribe"),  # OpenAI Whisper STT 모델 사용
        # stt=deepgram.STT(model="nova-2-general", language="ko"),
        # llm=openai.LLM(model="gpt-4o"),
        # llm=openai.realtime.RealtimeModel(model="gpt-4o-realtime-preview-2025-06-03"),
        # tts=openai.TTS(model="gpt-4o-mini-tts", voice="alloy"),  # 음성 기본 설정 
        # tts=elevenlabs.TTS(
        #         voice_id="tZarJVdIxWQ9lIXIV9qg",
        #         model="eleven_turbo_v2_5",
        #         voice_settings=elevenlabs.VoiceSettings(
        #             stability=0.5,
        #             similarity_boost=0.75,
        #             style=0.0,
        #             speed=1.0,
        #         ),
        #         encoding="mp3_44100_32",
        #     ),
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
        # transcription_enabled=False,   # 텍스트 전사 비활성화
        animation_enabled=True,       # 애니메이션 데이터 출력 활성화
        sync_transcription=False,
    )

    logger.info(f"애니메이션 데이터 스트리밍을 활성화했습니다. 대상: {participant.identity}")
    
    # Agent의 identity 로깅
    agent_identity = ctx.room.local_participant.identity
    logger.info(f"Agent Identity: {agent_identity}")

    # Agent 인스턴스 생성
    agent = FaceAgent(user_data, db)
    
    # 세션 시작
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=room_input_options,
        room_output_options=room_output_options
    )
    
    # 세션 종료 이벤트 핸들러 - 채팅 기록 저장
    @session.on("close")
    def on_session_close():
        """세션 종료 시 채팅 기록을 데이터베이스에 저장"""
        # agent.chat_ctx에서 현재 세션의 메시지들 가져오기
        chat_messages = []
        for item in agent.chat_ctx.items:
            if isinstance(item, llm.ChatMessage):
                # timestamp 변환 (float인 경우 datetime으로 변환)
                if isinstance(item.created_at, (int, float)):
                    timestamp = datetime.fromtimestamp(item.created_at)
                elif isinstance(item.created_at, datetime):
                    timestamp = item.created_at
                else:
                    timestamp = datetime.now()
                
                # content 변환 (리스트인 경우 텍스트만 추출)
                content_str = ""
                if isinstance(item.content, list):
                    content_str = " ".join(str(c) for c in item.content)
                else:
                    content_str = str(item.content)
                
                chat_messages.append(ChatMessage(
                    participant_id=participant.identity,
                    session_id=user_data.session_id,
                    timestamp=timestamp,
                    role=item.role,
                    content=content_str,
                    interrupted=getattr(item, 'interrupted', False)
                ))
        
        logger.info(f"세션 종료, 채팅 기록 저장 중... (총 {len(chat_messages)}개 메시지)")
        
        # 데이터베이스에 채팅 기록 저장
        if chat_messages:
            db.save_chat_messages(chat_messages)
            db.update_last_seen(participant.identity)
            
        # 사용자 요약 정보 로깅
        summary = db.get_user_summary(participant.identity)
        logger.info(f"사용자 정보: {summary}")
    
    # RPC 메서드 등록 - 사용자 주의 확인 메시지
    @ctx.room.local_participant.register_rpc_method("check_attention")
    async def check_attention(data: rtc.RpcInvocationData) -> str:
        """사용자의 주의를 환기시키는 RPC 메서드"""
        logger.info(f"RPC 'check_attention' 호출됨! 호출자: {data.caller_identity}")
        
        # Agent가 사용자에게 주의 환기 메시지를 음성으로 말하기
        attention_message = "너 지금 뭐해? 내 말 듣고 있어?"
        logger.info(f"Agent가 음성으로 말할 내용: {attention_message}")
        
        # session.say를 사용해서 즉시 음성으로 응답
        session.say(attention_message, allow_interruptions=True)
        
        logger.info(f"RPC 응답 완료: 주의 환기 메시지 전달")
        return "주의 환기 메시지를 음성으로 전달했습니다."
    
    logger.info("RPC 메서드 'check_attention' 등록 완료")
    

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    ) 