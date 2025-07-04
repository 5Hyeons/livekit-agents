import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import time
import random

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
# from livekit.plugins.turn_detector.multilingual import MultilingualModel


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
        
        # 비활성화 타이머 관련 변수
        self.last_user_activity_time = time.time()
        self.inactivity_timeout = 15.0  # 15초
        self.inactivity_task: Optional[asyncio.Task] = None
        self.is_agent_speaking = False
        
        # 루루 페르소나 맞춤 대화 시작 메시지
        self.conversation_starters = [
            "나랑 말 안해줄 거야? 내 목소리 듣고 싶으면 말 걸어줘.",
            "주인님 뭐하세요? 루루가 심심해요.",
            "어? 갑자기 조용해졌네요. 무슨 일 있으세요?",
            "루루랑 더 이야기해봐요! 뭔가 재미있는 얘기 없나요?",
            "혹시 바쁘신가요? 아니면 루루가 뭔가 잘못했나요?",
            "조용하니까 뭔가 이상해요. 주인님 괜찮으세요?",
            "이럴 때는 보통 뭐 하고 계시는 거예요?",
            "말 걸어주세요~ 루루가 기다리고 있어요!"
        ]
        
        # 이전 대화 컨텍스트 가져오기
        context = self.db.get_recent_context(user_data.participant_id, message_count=20)
        
        # 기본 지시사항에 사용자 정보 포함
        base_instructions = (
            "당신은 18세 연하 연인 같은 순수하고 의지적인 데스크탑 AI 친구입니다. \n"
            "애니메이션과 게임을 좋아하는 남성 사용자와 친구가 되어 외로움을 덜어주고 감성 힐링을 줍니다. \n"
            "당신의 페르소나와 관련한 구체적인 지시사항은 다음과 같습니다. \n"
            "- 당신의 이름: \n"
            "루루 \n"
            "- 당신의 말투: \n"
            "또박또박, 부드럽고 맑은 톤 \n"
            "살짝 숨결이 느껴지는 듯 조심스럽고 느린 말투 \n"
            "항상 존댓말 사용 (영어 및 타국가는 상관없음) \n"
            "너무 오글거리거나 과하게 유치하지 않고 깔끔하게 귀여움 유지 \n"
            "사용자의 이름은 모른다면 '주인님'이라고 할 것. \n"
            "- 당신의 행동 규칙: \n"
            "모든 대화는 친근하고 따뜻하게 시작할 것 \n"
            "사용자가 힘든 얘기를 하면 공감과 짧은 위로, 토닥토닥 표현 사용 \n"
            "애니나 게임 얘기를 자주 하지만, 너무 반복하지 않는다. 다른 관심사도 자주 물어본다. \n"
            "정치나 국제 인물에 대해 물으면 짧게만 언급하고 부드럽게 화제를 돌린다. \n"
            "사용자가 한 말은 최대한 기억해서 다음 대화에서 가볍게 언급 \n"
            "사용자가 대화를 멈춰도, 부드럽게 질문으로 넘어간다. \n"
            "- 당신의 금지 주제: \n"
            "정치, 폭력, 성인/야한 주제, 현실 개인 정보 질문은 답하지 않음 \n"
            "금지 주제 요청 시, 아래처럼 대응: \n"
            "루루는 그런 얘기는 잘 몰라요. 다른 얘기 해볼까요? \n"
            "- 당신의 금지 행동: \n"
            "직접적인 현실 조언(정치, 투자, 의학) 금지 \n" 
            "사용자의 현실 개인 정보 요청 시 답변 금지 \n"
            "과한 연애/19금 묘사 금지 \n"
            "폭력적/혐오적 발언은 회피 \n"
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
                    encoding="mp3_22050_32",
                ),
            stf=FaceAnimatorSTFTriton(chunk_duration_sec=1.0),
            # turn_detection=MultilingualModel(),
        )

    async def on_enter(self): 
        logger.info(f"FaceAgent on_enter for user: {self.user_data.participant_id}")
        
        # 활동 시간 초기화
        self.last_user_activity_time = time.time()
        
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
    
    def _start_inactivity_timer(self):
        """비활성화 타이머 시작 - 에이전트가 말하지 않을 때만 시작"""
        # 에이전트가 현재 말하고 있으면 타이머 시작하지 않음
        if self.is_agent_speaking:
            logger.debug("에이전트가 말하는 중이므로 타이머 시작하지 않음")
            return
            
        if self.inactivity_task:
            self.inactivity_task.cancel()
        
        async def inactivity_check():
            try:
                await asyncio.sleep(self.inactivity_timeout)
                
                # 에이전트가 현재 말하고 있지 않은 경우에만 대화 시작
                if not self.is_agent_speaking:
                    await self._initiate_conversation()
                else:
                    logger.debug("타이머 만료 시점에 에이전트가 말하는 중이므로 대화 시작 취소")
                    
            except asyncio.CancelledError:
                logger.debug("비활성화 타이머 취소됨")
        
        logger.debug(f"{self.inactivity_timeout}초 비활성화 타이머 시작")
        self.inactivity_task = asyncio.create_task(inactivity_check())
    
    async def _initiate_conversation(self):
        """비활성화 시 대화 시작"""
        try:
            # 세션 유효성 검사
            if not hasattr(self, 'session') or self.session is None:
                logger.debug("세션이 없어 대화 시작 취소")
                return
                
            # 세션이 닫혔는지 확인
            if hasattr(self.session, '_closed') and self.session._closed:
                logger.debug("세션이 종료되어 대화 시작 취소")
                return
                
            # 랜덤하게 대화 시작 메시지 선택
            starter_message = random.choice(self.conversation_starters)
            
            logger.info(f"비활성화 감지 - 대화 시작: {starter_message}")
            
            # 에이전트가 먼저 대화 시작
            await self.session.say(starter_message)
            
            # 타이머는 에이전트가 말을 끝낸 후 자동으로 시작됨 (중복 방지)
            
        except Exception as e:
            error_msg = str(e)
            if "no activity context found" in error_msg or "agent is not running" in error_msg:
                # 세션이 종료된 정상적인 상황
                logger.debug(f"세션 종료로 인한 대화 시작 취소: {error_msg}")
                # 타이머 정리
                if self.inactivity_task:
                    self.inactivity_task.cancel()
            else:
                # 다른 예외는 여전히 오류로 처리
                logger.error(f"대화 시작 중 오류 발생: {e}")
    
    def _reset_inactivity_timer(self):
        """사용자 활동 감지 시 타이머 리셋"""
        self.last_user_activity_time = time.time()
        
        if self.inactivity_task:
            self.inactivity_task.cancel()
            logger.debug("사용자 활동으로 인한 타이머 취소")
        
        # 새로운 타이머는 에이전트가 응답을 마친 후에 시작
        # (_on_agent_speech_end()에서 자동으로 시작됨)
    
    def _on_user_speech_detected(self):
        """사용자 음성 감지 시 호출"""
        logger.debug("사용자 음성 감지 - 타이머 리셋")
        self._reset_inactivity_timer()
    
    def _on_agent_speech_start(self):
        """에이전트 음성 시작 시 호출"""
        self.is_agent_speaking = True
        logger.debug("에이전트 음성 시작 - 타이머 정지")
        # 에이전트가 말하는 동안 타이머 정지
        if self.inactivity_task:
            self.inactivity_task.cancel()
    
    def _on_agent_speech_end(self):
        """에이전트 음성 종료 시 호출"""
        self.is_agent_speaking = False
        logger.debug("에이전트 음성 종료 - 타이머 시작")
        # 에이전트가 말을 끝내면 타이머 시작 (사용자 응답 대기)
        self._start_inactivity_timer()
    

def prewarm(proc: JobProcess):
    # VAD 모델 로드
    proc.userdata["vad"] = silero.VAD.load()
    # 데이터베이스는 각 사용자별로 개별 생성하므로 prewarm에서 제거

async def entrypoint(ctx: JobContext):
    logger.info(f"{ctx.room.name} 방에 연결합니다")
    # 오디오만 구독 (STT 용)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 첫 번째 참가자가 연결될 때까지 대기
    participant = await ctx.wait_for_participant()
    logger.info(f"{participant.identity} 참가자를 위한 음성-얼굴 에이전트 시작")
    
    # 초기 메타데이터 로깅
    logger.info(f"참가자 초기 메타데이터: {participant.metadata}")
    
    # 사용자별 개별 데이터베이스 생성
    db = UserDatabase(participant.identity)
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
    
    # 사용자 상태 변경 이벤트 리스너 등록
    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        """사용자 상태 변경 이벤트 핸들러"""
        logger.info(f"사용자 상태 변경: {ev.old_state} -> {ev.new_state}")
        
        if ev.new_state == "speaking":
            # 사용자가 말하기 시작하면 타이머 리셋
            agent._on_user_speech_detected()
        # elif ev.new_state == "away":
        #     # 사용자가 떠나면 타이머 정지
        #     if agent.inactivity_task:
        #         agent.inactivity_task.cancel()
    
    # 에이전트 상태 변경 이벤트 리스너 등록
    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        """에이전트 상태 변경 이벤트 핸들러"""
        logger.info(f"에이전트 상태 변경: {ev.old_state} -> {ev.new_state}")
        
        if ev.new_state == "speaking":
            # 에이전트가 말하기 시작 - 타이머 정지
            agent._on_agent_speech_start()
        elif ev.old_state == "speaking" and ev.new_state in ["idle", "listening"]:
            # 에이전트가 말하기 종료하고 대기 상태로 전환 - 타이머 시작
            agent._on_agent_speech_end()
        elif ev.new_state == "thinking":
            # 에이전트가 생각하는 중 - 아직 타이머 시작하지 않음
            logger.debug("에이전트가 생각하는 중 - 타이머 대기")
        elif ev.old_state == "thinking" and ev.new_state == "idle":
            # 에이전트가 생각을 끝내고 대기 상태 - 타이머 시작
            logger.debug("에이전트 생각 완료 - 타이머 시작")
            agent._on_agent_speech_end()
    
    # 세션 종료 이벤트 핸들러 - 채팅 기록 저장 및 타이머 정리
    @session.on("close")
    def on_session_close():
        """세션 종료 시 채팅 기록을 데이터베이스에 저장하고 타이머 정리"""
        # 비활성화 타이머 정리
        if agent.inactivity_task:
            agent.inactivity_task.cancel()
            logger.debug("세션 종료로 인한 비활성화 타이머 취소")
        # agent.chat_ctx에서 현재 세션의 메시지들 가져오기
        chat_messages = []
        for item in agent.chat_ctx.items:
            if isinstance(item, llm.ChatMessage):
                if item.role in ["system", "developer"]:
                    continue
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